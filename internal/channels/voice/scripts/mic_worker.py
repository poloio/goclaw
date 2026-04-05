#!/usr/bin/env python3
"""Persistent mic worker. Loads Silero VAD once, captures utterances on request.

Protocol (JSON lines on stdin/stdout):
  Ready:    → {"status": "ready"}
  Request:  ← {"out_path": "/tmp/foo.wav"}
  Response: → {"path": "/tmp/foo.wav"} or {"error": "no speech"} or {"error": "..."}
"""
import json
import signal
import struct
import sys
import time
import wave

import pyaudio
import torch


SAMPLE_RATE = 16000
VAD_THRESHOLD = 0.5
SILENCE_TIMEOUT = 0.8
CHUNK = 512


def find_input_device(pa):
    for i in range(pa.get_device_count()):
        if pa.get_device_info_by_index(i)["maxInputChannels"] > 0:
            return i
    return None


def main():
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

        stream = None
        try:
            req = json.loads(line)
            out_path = req["out_path"]

            # Re-check device in case it changed
            new_dev = find_input_device(pa)
            if new_dev is not None:
                dev = new_dev

            stream = pa.open(format=pyaudio.paInt16, channels=1, rate=SAMPLE_RATE,
                             input=True, input_device_index=dev, frames_per_buffer=CHUNK)

            frames = []
            silence_start = None
            has_speech = False

            while True:
                data = stream.read(CHUNK, exception_on_overflow=False)
                frames.append(data)
                n_samples = len(data) // 2
                if n_samples == 0:
                    continue
                samples = struct.unpack(f"<{n_samples}h", data)
                tensor = torch.FloatTensor(samples) / 32768.0

                with torch.no_grad():
                    prob = model(tensor, SAMPLE_RATE).item()

                if prob > VAD_THRESHOLD:
                    has_speech = True
                    silence_start = None
                elif has_speech:
                    if silence_start is None:
                        silence_start = time.time()
                    elif time.time() - silence_start > SILENCE_TIMEOUT:
                        break

                if len(frames) * CHUNK / SAMPLE_RATE > 30:
                    break

            model.reset_states()

            if not has_speech:
                print(json.dumps({"error": "no speech"}), flush=True)
                continue

            with wave.open(out_path, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(SAMPLE_RATE)
                wf.writeframes(b"".join(frames))

            print(json.dumps({"path": out_path}), flush=True)

        except Exception as e:
            print(json.dumps({"error": str(e)}), flush=True)
        finally:
            if stream is not None:
                try:
                    stream.stop_stream()
                    stream.close()
                except Exception:
                    pass

    pa.terminate()


if __name__ == "__main__":
    signal.signal(signal.SIGINT, lambda *_: sys.exit(0))
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    main()
