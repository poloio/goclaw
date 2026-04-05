#!/usr/bin/env python3
"""Persistent wake word + mic worker. Loads openWakeWord and Silero VAD once.

Phase 1: listen for wake word (openWakeWord, low CPU)
Phase 2: capture utterance (Silero VAD, same audio stream — no reopen)

Protocol (JSON lines on stdin/stdout):
  Ready:    → {"status": "ready"}
  Request:  ← {"out_path": "/tmp/foo.wav"}
  Response: → {"path": "/tmp/foo.wav"} or {"error": "no speech"} or {"error": "..."}
"""
import json
import struct
import sys
import time
import wave

import pyaudio
import torch
from openwakeword.model import Model as OWWModel


def find_input_device(pa):
    for i in range(pa.get_device_count()):
        if pa.get_device_info_by_index(i)["maxInputChannels"] > 0:
            return i
    return None


def main():
    wake_word = sys.argv[1] if len(sys.argv) > 1 else "hey_jarvis"
    sample_rate = 16000
    vad_threshold = 0.5
    silence_timeout = 1.0
    ww_threshold = 0.5
    # Use 1280 samples (80ms) for both phases — openWakeWord needs 80ms,
    # Silero VAD works fine with any chunk size at 16kHz.
    chunk = 1280

    oww = OWWModel(wakeword_models=[wake_word], inference_framework="onnx")
    vad, _ = torch.hub.load("snakers4/silero-vad", "silero_vad", trust_repo=True)

    pa = pyaudio.PyAudio()
    dev = find_input_device(pa)
    if dev is None:
        print(json.dumps({"error": "no microphone found"}), flush=True)
        sys.exit(1)

    print(json.dumps({"status": "ready"}), flush=True)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            out_path = req["out_path"]

            stream = pa.open(format=pyaudio.paInt16, channels=1, rate=sample_rate,
                             input=True, input_device_index=dev, frames_per_buffer=chunk)

            # --- Phase 1: wait for wake word (same stream used for phase 2) ---
            ww_detected = False
            while True:
                data = stream.read(chunk, exception_on_overflow=False)
                n_samples = len(data) // 2
                samples = struct.unpack(f"<{n_samples}h", data)
                audio_f32 = [s / 32768.0 for s in samples]

                prediction = oww.predict_clip(audio_f32)
                for name, scores in prediction.items():
                    if max(scores) > ww_threshold:
                        ww_detected = True
                        break
                if ww_detected:
                    sys.stderr.write(f"[WW] '{wake_word}' detected\n")
                    break

            # --- Phase 2: capture utterance (same stream, no reopen) ---
            frames = []
            silence_start = None
            has_speech = False

            while True:
                data = stream.read(chunk, exception_on_overflow=False)
                frames.append(data)
                n_samples = len(data) // 2
                if n_samples == 0:
                    continue
                samples = struct.unpack(f"<{n_samples}h", data)
                tensor = torch.FloatTensor(samples) / 32768.0
                prob = vad(tensor, sample_rate).item()

                if prob > vad_threshold:
                    has_speech = True
                    silence_start = None
                elif has_speech:
                    if silence_start is None:
                        silence_start = time.time()
                    elif time.time() - silence_start > silence_timeout:
                        break
                else:
                    # Grace period: 5s with no speech after wake word
                    if len(frames) * chunk / sample_rate > 5:
                        break

                if len(frames) * chunk / sample_rate > 30:
                    break

            stream.stop_stream()
            stream.close()
            vad.reset_states()
            oww.reset()

            if not has_speech:
                print(json.dumps({"error": "no speech after wake word"}), flush=True)
                continue

            with wave.open(out_path, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(sample_rate)
                wf.writeframes(b"".join(frames))

            print(json.dumps({"path": out_path}), flush=True)

        except Exception as e:
            print(json.dumps({"error": str(e)}), flush=True)


if __name__ == "__main__":
    main()
