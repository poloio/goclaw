package voice

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"os"
	"os/exec"
	"path/filepath"
	"sync"
)

// worker manages a persistent Python subprocess that communicates via
// JSON-lines on stdin/stdout. Models load once at startup and stay resident.
type worker struct {
	cmd    *exec.Cmd
	stdin  *json.Encoder
	stdout *bufio.Scanner
	mu     sync.Mutex // serializes request/response pairs
}

// findPython returns the jarvis-env venv python if available, else system python3.
func findPython() string {
	venv := filepath.Join(os.Getenv("HOME"), "jarvis-env", "bin", "python3")
	if _, err := os.Stat(venv); err == nil {
		return venv
	}
	return "python3"
}

// startWorker launches a persistent Python script as a subprocess.
// scriptPath is the path to a .py file (written to a temp dir from embedded content).
// args are extra CLI arguments passed to the script.
func startWorker(ctx context.Context, python, scriptPath string, args ...string) (*worker, error) {
	cmdArgs := append([]string{"-u", scriptPath}, args...) // -u for unbuffered stdout
	cmd := exec.CommandContext(ctx, python, cmdArgs...)
	cmd.Stderr = os.Stderr

	stdinPipe, err := cmd.StdinPipe()
	if err != nil {
		return nil, fmt.Errorf("worker stdin pipe: %w", err)
	}
	stdoutPipe, err := cmd.StdoutPipe()
	if err != nil {
		return nil, fmt.Errorf("worker stdout pipe: %w", err)
	}

	if err := cmd.Start(); err != nil {
		return nil, fmt.Errorf("worker start: %w", err)
	}

	w := &worker{
		cmd:    cmd,
		stdin:  json.NewEncoder(stdinPipe),
		stdout: bufio.NewScanner(stdoutPipe),
	}

	// Wait for ready signal
	if w.stdout.Scan() {
		var msg map[string]string
		if err := json.Unmarshal(w.stdout.Bytes(), &msg); err == nil {
			if msg["status"] == "ready" {
				slog.Info("worker ready", "script", scriptPath)
				return w, nil
			}
			if errMsg := msg["error"]; errMsg != "" {
				cmd.Process.Kill()
				return nil, fmt.Errorf("worker init error: %s", errMsg)
			}
		}
	}
	if err := w.stdout.Err(); err != nil {
		cmd.Process.Kill()
		return nil, fmt.Errorf("worker stdout: %w", err)
	}

	cmd.Process.Kill()
	return nil, fmt.Errorf("worker did not send ready signal")
}

// call sends a JSON request and reads a JSON response. Thread-safe.
func (w *worker) call(req any, resp any) error {
	w.mu.Lock()
	defer w.mu.Unlock()

	if err := w.stdin.Encode(req); err != nil {
		return fmt.Errorf("worker send: %w", err)
	}

	if !w.stdout.Scan() {
		if err := w.stdout.Err(); err != nil {
			return fmt.Errorf("worker recv: %w", err)
		}
		return fmt.Errorf("worker: subprocess closed stdout")
	}

	if err := json.Unmarshal(w.stdout.Bytes(), resp); err != nil {
		return fmt.Errorf("worker parse: %w (raw: %s)", err, w.stdout.Text())
	}
	return nil
}

// stop gracefully shuts down the worker.
func (w *worker) stop() {
	if w == nil || w.cmd == nil || w.cmd.Process == nil {
		return
	}
	w.cmd.Process.Signal(os.Interrupt)
	w.cmd.Wait()
}
