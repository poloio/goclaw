package voice

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
)

// PiperSynthesizer uses Piper TTS (subprocess) for speech synthesis.
type PiperSynthesizer struct {
	VoicePath  string // path to .onnx model
	ConfigPath string // path to .onnx.json config
	sampleRate int
}

// NewPiperSynthesizer creates a synthesizer that auto-discovers a Piper voice.
// preferred is an optional voice name to try first (e.g. "es_ES-davefx-medium").
// candidates are tried in order if preferred is empty or not found.
func NewPiperSynthesizer(preferred string, candidates []string) (*PiperSynthesizer, error) {
	voicesDir := filepath.Join(os.Getenv("HOME"), ".local", "share", "piper", "voices")

	if preferred != "" {
		candidates = append([]string{preferred}, candidates...)
	}
	if len(candidates) == 0 {
		candidates = []string{"es_ES-davefx-medium", "es_MX-claude-high"}
	}

	for _, name := range candidates {
		onnx := filepath.Join(voicesDir, name+".onnx")
		cfg := filepath.Join(voicesDir, name+".onnx.json")
		if _, err := os.Stat(onnx); err != nil {
			continue
		}
		if _, err := os.Stat(cfg); err != nil {
			continue
		}

		sr := 22050
		if data, err := os.ReadFile(cfg); err == nil {
			var parsed struct {
				Audio struct {
					SampleRate int `json:"sample_rate"`
				} `json:"audio"`
			}
			if json.Unmarshal(data, &parsed) == nil && parsed.Audio.SampleRate > 0 {
				sr = parsed.Audio.SampleRate
			}
		}

		slog.Info("piper TTS voice found", "voice", name, "sample_rate", sr)
		return &PiperSynthesizer{VoicePath: onnx, ConfigPath: cfg, sampleRate: sr}, nil
	}

	return nil, fmt.Errorf("no piper voice found in %s (tried %v)", voicesDir, candidates)
}

func (p *PiperSynthesizer) Synthesize(ctx context.Context, text string) ([]byte, error) {
	tmp, err := os.CreateTemp("", "voice-tts-*.wav")
	if err != nil {
		return nil, err
	}
	tmpPath := tmp.Name()
	tmp.Close()
	defer os.Remove(tmpPath)

	cmd := exec.CommandContext(ctx, "piper",
		"--model", p.VoicePath,
		"--config", p.ConfigPath,
		"--output_file", tmpPath,
	)
	cmd.Stdin = strings.NewReader(text)

	if out, err := cmd.CombinedOutput(); err != nil {
		return nil, fmt.Errorf("piper error: %s: %w", string(out), err)
	}

	return os.ReadFile(tmpPath)
}

func (p *PiperSynthesizer) SampleRate() int {
	return p.sampleRate
}
