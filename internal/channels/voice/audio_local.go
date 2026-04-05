package voice

import (
	"bytes"
	"context"
	"fmt"
	"os"
	"os/exec"
)

// LocalMic captures audio using a persistent Python worker with Silero VAD.
// Models load once at startup. Each ListenOnce() call sends a request and waits.
type LocalMic struct {
	w *worker
}

func NewLocalMic() *LocalMic {
	return &LocalMic{}
}

// Start launches the persistent mic worker.
func (m *LocalMic) Start(ctx context.Context) error {
	script, err := getScript("mic_worker.py")
	if err != nil {
		return err
	}
	w, err := startWorker(ctx, findPython(), script)
	if err != nil {
		return fmt.Errorf("mic worker start: %w", err)
	}
	m.w = w
	return nil
}

func (m *LocalMic) ListenOnce(ctx context.Context) (string, error) {
	if m.w == nil {
		return "", fmt.Errorf("mic worker not started")
	}

	tmp, err := os.CreateTemp("", "voice-mic-*.wav")
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
		return "", fmt.Errorf("mic call: %w", err)
	}
	if resp.Error != "" {
		os.Remove(wavPath)
		return "", nil // no speech, not an error
	}
	return resp.Path, nil
}

func (m *LocalMic) Close() error {
	if m.w != nil {
		m.w.stop()
	}
	return nil
}

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
	cmd.Stdin = bytes.NewReader(wavData)
	if out, err := cmd.CombinedOutput(); err != nil {
		return fmt.Errorf("aplay: %s: %w", string(out), err)
	}
	return nil
}

func (s *LocalSpeaker) Close() error { return nil }
