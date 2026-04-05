package voice

import (
	"context"
	"fmt"
	"log/slog"
	"os"
)

// WakeWordMic uses a persistent Python worker running openWakeWord + Silero VAD.
// Models load once at startup. Same audio stream for both phases (no gap).
type WakeWordMic struct {
	WakeWord string
	python   string
	w        *worker
}

func NewWakeWordMic(wakeWord string) *WakeWordMic {
	if wakeWord == "" {
		wakeWord = "hey_jarvis"
	}
	return &WakeWordMic{WakeWord: wakeWord, python: findPython()}
}

func (m *WakeWordMic) Start(ctx context.Context) error {
	return m.startWorker(ctx)
}

func (m *WakeWordMic) startWorker(ctx context.Context) error {
	script, err := getScript("wakeword_worker.py")
	if err != nil {
		return err
	}
	w, err := startWorker(ctx, m.python, script, m.WakeWord)
	if err != nil {
		return fmt.Errorf("wake word worker start: %w", err)
	}
	m.w = w
	return nil
}

func (m *WakeWordMic) ListenOnce(ctx context.Context) (string, error) {
	// Auto-restart dead worker
	if m.w == nil || !m.w.alive() {
		slog.Warn("wake word worker dead, restarting")
		if err := m.startWorker(ctx); err != nil {
			return "", fmt.Errorf("wake word restart: %w", err)
		}
	}

	tmp, err := os.CreateTemp("", "voice-ww-*.wav")
	if err != nil {
		return "", err
	}
	wavPath := tmp.Name()
	tmp.Close()

	req := map[string]string{"out_path": wavPath}
	var resp struct {
		Path  string `json:"path"`
		Error string `json:"error"`
	}
	if err := m.w.call(ctx, req, &resp); err != nil {
		os.Remove(wavPath)
		return "", fmt.Errorf("wake word call: %w", err)
	}
	if resp.Error != "" {
		os.Remove(wavPath)
		return "", nil
	}
	return resp.Path, nil
}

func (m *WakeWordMic) Close() error {
	if m.w != nil {
		m.w.stop()
	}
	return nil
}
