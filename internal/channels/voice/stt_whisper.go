package voice

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
)

// WhisperTranscriber uses faster-whisper (Python subprocess) for STT.
type WhisperTranscriber struct {
	Model    string // whisper model size: "tiny", "base", "small", "medium"
	Language string // language code: "es", "en", etc.
	Python   string // path to python3 binary
}

// NewWhisperTranscriber creates a transcriber using faster-whisper.
// It auto-detects the jarvis-env venv python if available.
func NewWhisperTranscriber(model, language string) *WhisperTranscriber {
	if model == "" {
		model = "base"
	}
	if language == "" {
		language = "es"
	}

	python := "python3"
	venvPython := filepath.Join(os.Getenv("HOME"), "jarvis-env", "bin", "python3")
	if _, err := os.Stat(venvPython); err == nil {
		python = venvPython
	}

	return &WhisperTranscriber{Model: model, Language: language, Python: python}
}

func (w *WhisperTranscriber) Transcribe(ctx context.Context, wavPath string) (string, error) {
	script := fmt.Sprintf(`
import sys, json
from faster_whisper import WhisperModel
model = WhisperModel("%s", device="cpu", compute_type="int8")
segments, info = model.transcribe(sys.argv[1], language="%s", beam_size=1, vad_filter=True)
text = " ".join(s.text.strip() for s in segments)
print(json.dumps({"text": text.strip()}))
`, w.Model, w.Language)

	cmd := exec.CommandContext(ctx, w.Python, "-c", script, wavPath)
	out, err := cmd.Output()
	if err != nil {
		if exitErr, ok := err.(*exec.ExitError); ok {
			return "", fmt.Errorf("whisper STT error: %s", string(exitErr.Stderr))
		}
		return "", fmt.Errorf("whisper STT exec: %w", err)
	}

	var result struct {
		Text string `json:"text"`
	}
	if err := json.Unmarshal(out, &result); err != nil {
		return "", fmt.Errorf("whisper STT parse: %w", err)
	}
	return result.Text, nil
}
