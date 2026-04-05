#!/usr/bin/env python3
"""Persistent mic worker. Loads Silero VAD once, captures utterances on request.

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


def find_input_device(pa):
    for i in range(pa.get_device_count()):
        if pa.get_device_info_by_index(i)["maxInputChannels"] > 0:
            return i
    return None


def main():
    sample_rate = 16000
    vad_threshold = 0.5
    silence_timeout = 0.8
    chunk = 512

    model, _ = torch.hub.load("snakers4/silero-vad", "silero_vad", trust_repo=True)

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
                prob = model(tensor, sample_rate).item()

                if prob > vad_threshold:
                    has_speech = True
                    silence_start = None
                elif has_speech:
                    if silence_start is None:
                        silence_start = time.time()
                    elif time.time() - silence_start > silence_timeout:
                        break

                if len(frames) * chunk / sample_rate > 30:
                    break

            stream.stop_stream()
            stream.close()
            model.reset_states()

            if not has_speech:
                print(json.dumps({"error": "no speech"}), flush=True)
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
