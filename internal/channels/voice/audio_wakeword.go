package voice

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"os"
	"os/exec"
	"path/filepath"
)

// WakeWordMic wraps local mic capture with wake word gating.
// It continuously listens for a wake word (via openWakeWord), and only
// when detected does it switch to Silero VAD to capture the full utterance.
//
// This prevents the pipeline from processing background noise, conversations
// not directed at the assistant, or car audio (music, navigation prompts).
type WakeWordMic struct {
	Python         string
	WakeWord       string  // wake word model name, e.g. "hey_jarvis", "alexa", or path to .onnx
	SampleRate     int
	VadThreshold   float64
	SilenceTimeout float64
}

// NewWakeWordMic creates a wake-word-gated microphone source.
// wakeWord can be a built-in model name ("hey_jarvis", "alexa", "hey_mycroft", etc.)
// or a path to a custom .onnx/.tflite model file.
func NewWakeWordMic(wakeWord string) *WakeWordMic {
	if wakeWord == "" {
		wakeWord = "hey_jarvis"
	}
	python := "python3"
	venvPython := filepath.Join(os.Getenv("HOME"), "jarvis-env", "bin", "python3")
	if _, err := os.Stat(venvPython); err == nil {
		python = venvPython
	}
	return &WakeWordMic{
		Python:         python,
		WakeWord:       wakeWord,
		SampleRate:     16000,
		VadThreshold:   0.5,
		SilenceTimeout: 1.0,
	}
}

// ListenOnce blocks until the wake word is heard, then captures the following
// utterance via Silero VAD. Returns the path to a WAV file containing the
// utterance (without the wake word). Caller must delete the file.
func (m *WakeWordMic) ListenOnce(ctx context.Context) (string, error) {
	tmp, err := os.CreateTemp("", "voice-ww-*.wav")
	if err != nil {
		return "", err
	}
	wavPath := tmp.Name()
	tmp.Close()

	// Single Python process handles both wake word detection and VAD capture.
	// Phase 1: listen for wake word (low CPU, openWakeWord)
	// Phase 2: capture utterance (Silero VAD)
	script := fmt.Sprintf(`
import sys, json, struct, time, wave
import pyaudio, torch
import openwakeword
from openwakeword.model import Model as OWWModel

SAMPLE_RATE = %d
VAD_THRESHOLD = %f
SILENCE_TIMEOUT = %f
VAD_CHUNK = 512
OWW_CHUNK = 1280  # openWakeWord expects 80ms chunks at 16kHz
WAKE_WORD = "%s"
OUT_PATH = sys.argv[1]
WW_THRESHOLD = 0.5

# --- Phase 0: Setup ---
oww_model = OWWModel(wakeword_models=[WAKE_WORD], inference_framework="onnx")
vad_model, _ = torch.hub.load("snakers4/silero-vad", "silero_vad", trust_repo=True)

pa = pyaudio.PyAudio()
dev = None
for i in range(pa.get_device_count()):
    if pa.get_device_info_by_index(i)["maxInputChannels"] > 0:
        dev = i
        break
if dev is None:
    pa.terminate()
    print(json.dumps({"error": "no microphone"}))
    sys.exit(0)

stream = pa.open(format=pyaudio.paInt16, channels=1, rate=SAMPLE_RATE,
                 input=True, input_device_index=dev, frames_per_buffer=OWW_CHUNK)

# --- Phase 1: Wait for wake word ---
sys.stderr.write("[WW] Listening for '%s'...\n" %% WAKE_WORD)
try:
    while True:
        data = stream.read(OWW_CHUNK, exception_on_overflow=False)
        audio_i16 = struct.unpack("<%dh" %% OWW_CHUNK, data)
        audio_f32 = [s / 32768.0 for s in audio_i16]

        prediction = oww_model.predict_clip(audio_f32)
        # oww returns dict of model_name -> score
        for name, scores in prediction.items():
            if max(scores) > WW_THRESHOLD:
                sys.stderr.write("[WW] Wake word detected!\n")
                # Fall through to phase 2
                raise StopIteration()
except StopIteration:
    pass

# --- Phase 2: Capture utterance via Silero VAD ---
sys.stderr.write("[VAD] Capturing utterance...\n")
# Reopen stream with smaller chunk for VAD
stream.stop_stream()
stream.close()
stream = pa.open(format=pyaudio.paInt16, channels=1, rate=SAMPLE_RATE,
                 input=True, input_device_index=dev, frames_per_buffer=VAD_CHUNK)

frames = []
silence_start = None
has_speech = False

while True:
    data = stream.read(VAD_CHUNK, exception_on_overflow=False)
    frames.append(data)
    samples = struct.unpack("<%dh" %% VAD_CHUNK, data)
    tensor = torch.FloatTensor(samples) / 32768.0
    prob = vad_model(tensor, SAMPLE_RATE).item()

    if prob > VAD_THRESHOLD:
        has_speech = True
        silence_start = None
    elif has_speech:
        if silence_start is None:
            silence_start = time.time()
        elif time.time() - silence_start > SILENCE_TIMEOUT:
            break
    else:
        # No speech yet after wake word — allow a few seconds grace period
        if len(frames) * VAD_CHUNK / SAMPLE_RATE > 5:
            break  # 5s with no speech after wake word, give up

    if len(frames) * VAD_CHUNK / SAMPLE_RATE > 30:
        break  # safety cap

stream.stop_stream()
stream.close()
pa.terminate()
vad_model.reset_states()

if not has_speech:
    print(json.dumps({"error": "no speech after wake word"}))
    sys.exit(0)

with wave.open(OUT_PATH, "wb") as wf:
    wf.setnchannels(1)
    wf.setsampwidth(2)
    wf.setframerate(SAMPLE_RATE)
    wf.writeframes(b"".join(frames))

print(json.dumps({"path": OUT_PATH}))
`, m.SampleRate, m.VadThreshold, m.SilenceTimeout, m.WakeWord)

	cmd := exec.CommandContext(ctx, m.Python, "-c", script, wavPath)
	cmd.Stderr = os.Stderr
	out, err := cmd.Output()
	if err != nil {
		os.Remove(wavPath)
		if exitErr, ok := err.(*exec.ExitError); ok {
			return "", fmt.Errorf("wake word mic error: %s", string(exitErr.Stderr))
		}
		return "", fmt.Errorf("wake word mic: %w", err)
	}

	var result struct {
		Path  string `json:"path"`
		Error string `json:"error"`
	}
	if err := json.Unmarshal(out, &result); err != nil {
		os.Remove(wavPath)
		return "", fmt.Errorf("wake word mic parse: %w", err)
	}
	if result.Error != "" {
		os.Remove(wavPath)
		slog.Debug("wake word mic: no utterance", "reason", result.Error)
		return "", nil
	}
	return result.Path, nil
}

func (m *WakeWordMic) Close() error { return nil }
