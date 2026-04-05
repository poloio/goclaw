package voice

import (
	"context"
	"fmt"
	"os"
)

// WakeWordMic uses a persistent Python worker running openWakeWord + Silero VAD.
// Models load once at startup. The worker listens for the wake word, then captures
// the utterance — all on the same audio stream (no reopen gap).
type WakeWordMic struct {
	WakeWord string
	w        *worker
}

func NewWakeWordMic(wakeWord string) *WakeWordMic {
	if wakeWord == "" {
		wakeWord = "hey_jarvis"
	}
	return &WakeWordMic{WakeWord: wakeWord}
}

// Start launches the persistent wake word worker.
func (m *WakeWordMic) Start(ctx context.Context) error {
	script, err := getScript("wakeword_worker.py")
	if err != nil {
		return err
	}
	w, err := startWorker(ctx, findPython(), script, m.WakeWord)
	if err != nil {
		return fmt.Errorf("wake word worker start: %w", err)
	}
	m.w = w
	return nil
}

func (m *WakeWordMic) ListenOnce(ctx context.Context) (string, error) {
	if m.w == nil {
		return "", fmt.Errorf("wake word worker not started")
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
	if err := m.w.call(req, &resp); err != nil {
		os.Remove(wavPath)
		return "", fmt.Errorf("wake word call: %w", err)
	}
	if resp.Error != "" {
		os.Remove(wavPath)
		return "", nil // no speech after wake word, not an error
	}
	return resp.Path, nil
}

func (m *WakeWordMic) Close() error {
	if m.w != nil {
		m.w.stop()
	}
	return nil
}
