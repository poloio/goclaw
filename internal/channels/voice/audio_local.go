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

// LocalMic captures audio from a local ALSA microphone using a Python helper
// (PyAudio + Silero VAD). This is the primary audio source for on-device deployment.
type LocalMic struct {
	Python     string // path to python3 with pyaudio+torch
	SampleRate int
	VadThreshold float64
	SilenceTimeout float64
}

// NewLocalMic creates a local microphone source.
func NewLocalMic() *LocalMic {
	python := "python3"
	venvPython := filepath.Join(os.Getenv("HOME"), "jarvis-env", "bin", "python3")
	if _, err := os.Stat(venvPython); err == nil {
		python = venvPython
	}
	return &LocalMic{
		Python:         python,
		SampleRate:     16000,
		VadThreshold:   0.5,
		SilenceTimeout: 0.8,
	}
}

// ListenOnce blocks until an utterance is detected via Silero VAD, then returns
// the path to a temporary WAV file. Caller must delete the file after use.
func (m *LocalMic) ListenOnce(ctx context.Context) (string, error) {
	tmp, err := os.CreateTemp("", "voice-mic-*.wav")
	if err != nil {
		return "", err
	}
	wavPath := tmp.Name()
	tmp.Close()

	script := fmt.Sprintf(`
import sys, json, struct, time, wave, tempfile
import pyaudio, torch

SAMPLE_RATE = %d
VAD_THRESHOLD = %f
SILENCE_TIMEOUT = %f
CHUNK = 512
OUT_PATH = sys.argv[1]

model, utils = torch.hub.load("snakers4/silero-vad", "silero_vad", trust_repo=True)
pa = pyaudio.PyAudio()

# Find first input device
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
                 input=True, input_device_index=dev, frames_per_buffer=CHUNK)

frames = []
silence_start = None
has_speech = False

try:
    while True:
        data = stream.read(CHUNK, exception_on_overflow=False)
        frames.append(data)
        samples = struct.unpack("<%dh" %% CHUNK, data)
        tensor = torch.FloatTensor(samples) / 32768.0
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
finally:
    stream.stop_stream()
    stream.close()
    pa.terminate()
    model.reset_states()

if not has_speech:
    print(json.dumps({"error": "no speech"}))
    sys.exit(0)

with wave.open(OUT_PATH, "wb") as wf:
    wf.setnchannels(1)
    wf.setsampwidth(2)
    wf.setframerate(SAMPLE_RATE)
    wf.writeframes(b"".join(frames))

print(json.dumps({"path": OUT_PATH}))
`, m.SampleRate, m.VadThreshold, m.SilenceTimeout)

	cmd := exec.CommandContext(ctx, m.Python, "-c", script, wavPath)
	cmd.Stderr = os.Stderr // show VAD/mic errors
	out, err := cmd.Output()
	if err != nil {
		os.Remove(wavPath)
		if exitErr, ok := err.(*exec.ExitError); ok {
			return "", fmt.Errorf("mic capture error: %s", string(exitErr.Stderr))
		}
		return "", fmt.Errorf("mic capture: %w", err)
	}

	var result struct {
		Path  string `json:"path"`
		Error string `json:"error"`
	}
	if err := json.Unmarshal(out, &result); err != nil {
		os.Remove(wavPath)
		return "", fmt.Errorf("mic parse: %w", err)
	}
	if result.Error != "" {
		os.Remove(wavPath)
		slog.Debug("mic: no utterance", "reason", result.Error)
		return "", nil // no speech, not an error
	}
	return result.Path, nil
}

func (m *LocalMic) Close() error { return nil }

// LocalSpeaker plays audio through a local ALSA output device using aplay.
type LocalSpeaker struct {
	Device string // ALSA device (e.g. "default", "hw:1,0"). Empty = default.
}

func NewLocalSpeaker(device string) *LocalSpeaker {
	return &LocalSpeaker{Device: device}
}

func (s *LocalSpeaker) Play(ctx context.Context, wavData []byte) error {
	args := []string{"-q", "-t", "wav", "-"}
	if s.Device != "" {
		args = append([]string{"-D", s.Device}, args...)
	}
	cmd := exec.CommandContext(ctx, "aplay", args...)
	cmd.Stdin = bytesReader(wavData)
	if out, err := cmd.CombinedOutput(); err != nil {
		return fmt.Errorf("aplay error: %s: %w", string(out), err)
	}
	return nil
}

func (s *LocalSpeaker) Close() error { return nil }

type bytesReaderCloser struct{ *os.File }

// bytesReader wraps []byte for use as cmd.Stdin via a temp pipe.
func bytesReader(data []byte) *readerWrapper {
	return &readerWrapper{data: data, pos: 0}
}

type readerWrapper struct {
	data []byte
	pos  int
}

func (r *readerWrapper) Read(p []byte) (int, error) {
	if r.pos >= len(r.data) {
		return 0, fmt.Errorf("EOF")
	}
	n := copy(p, r.data[r.pos:])
	r.pos += n
	return n, nil
}
