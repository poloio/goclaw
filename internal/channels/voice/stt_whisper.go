package voice

import (
	"context"
	"fmt"
	"log/slog"
)

// WhisperTranscriber uses a persistent faster-whisper Python worker for STT.
// The model loads once at startup and stays resident across all transcriptions.
type WhisperTranscriber struct {
	Model    string
	Language string
	python   string
	w        *worker
}

func NewWhisperTranscriber(model, language string) *WhisperTranscriber {
	if model == "" {
		model = "base"
	}
	if language == "" {
		language = "es"
	}
	return &WhisperTranscriber{Model: model, Language: language, python: findPython()}
}

func (t *WhisperTranscriber) Start(ctx context.Context) error {
	return t.startWorker(ctx)
}

func (t *WhisperTranscriber) startWorker(ctx context.Context) error {
	script, err := getScript("stt_worker.py")
	if err != nil {
		return err
	}
	w, err := startWorker(ctx, t.python, script, t.Model, t.Language)
	if err != nil {
		return fmt.Errorf("STT worker start: %w", err)
	}
	t.w = w
	return nil
}

func (t *WhisperTranscriber) Transcribe(ctx context.Context, wavPath string) (string, error) {
	// Auto-restart dead worker
	if t.w == nil || !t.w.alive() {
		slog.Warn("STT worker dead, restarting")
		if err := t.startWorker(ctx); err != nil {
			return "", fmt.Errorf("STT restart: %w", err)
		}
	}

	req := map[string]string{"wav_path": wavPath}
	var resp struct {
		Text  string `json:"text"`
		Error string `json:"error"`
	}
	if err := t.w.call(ctx, req, &resp); err != nil {
		return "", fmt.Errorf("STT call: %w", err)
	}
	if resp.Error != "" {
		return "", fmt.Errorf("STT: %s", resp.Error)
	}
	return resp.Text, nil
}

func (t *WhisperTranscriber) Stop() {
	if t.w != nil {
		t.w.stop()
	}
}
