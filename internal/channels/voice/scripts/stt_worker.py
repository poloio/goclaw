#!/usr/bin/env python3
"""Persistent STT worker. Loads faster-whisper once, transcribes WAV files on request.

Protocol (JSON lines on stdin/stdout):
  Ready:    → {"status": "ready"}
  Request:  ← {"wav_path": "/tmp/foo.wav"}
  Response: → {"text": "transcribed text"} or {"error": "..."}
"""
import json
import sys

def main():
    model_size = sys.argv[1] if len(sys.argv) > 1 else "base"
    language = sys.argv[2] if len(sys.argv) > 2 else "es"

    from faster_whisper import WhisperModel
    model = WhisperModel(model_size, device="cpu", compute_type="int8")

    print(json.dumps({"status": "ready"}), flush=True)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            wav_path = req["wav_path"]
            segments, info = model.transcribe(wav_path, language=language, beam_size=1, vad_filter=True)
            text = " ".join(s.text.strip() for s in segments).strip()
            print(json.dumps({"text": text}), flush=True)
        except Exception as e:
            print(json.dumps({"error": str(e)}), flush=True)


if __name__ == "__main__":
    main()
