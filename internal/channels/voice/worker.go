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
	"strings"
	"sync"
	"time"
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

// startWorker launches a persistent Python script and waits for the ready signal.
// Tolerates noisy stdout (library warnings) before the JSON ready message.
// Times out after 120s if the ready signal never arrives.
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

	// Wait for ready signal with timeout. Skip non-JSON lines (library warnings).
	deadline := time.After(120 * time.Second)
	readyCh := make(chan error, 1)
	go func() {
		for w.stdout.Scan() {
			line := w.stdout.Text()
			// Skip non-JSON lines (torch/onnx warnings printed to stdout)
			if !strings.HasPrefix(strings.TrimSpace(line), "{") {
				slog.Debug("worker: skipping non-JSON line", "line", line)
				continue
			}
			var msg map[string]string
			if err := json.Unmarshal([]byte(line), &msg); err != nil {
				slog.Debug("worker: skipping unparseable line", "line", line)
				continue
			}
			if msg["status"] == "ready" {
				readyCh <- nil
				return
			}
			if errMsg := msg["error"]; errMsg != "" {
				readyCh <- fmt.Errorf("worker init: %s", errMsg)
				return
			}
		}
		readyCh <- fmt.Errorf("worker stdout closed before ready (err: %v)", w.stdout.Err())
	}()

	select {
	case err := <-readyCh:
		if err != nil {
			cmd.Process.Kill()
			return nil, err
		}
		slog.Info("worker ready", "script", scriptPath)
		return w, nil
	case <-deadline:
		cmd.Process.Kill()
		return nil, fmt.Errorf("worker startup timeout (120s)")
	case <-ctx.Done():
		cmd.Process.Kill()
		return nil, ctx.Err()
	}
}

// call sends a JSON request and reads a JSON response. Thread-safe.
// Respects the provided context — returns early if cancelled or timed out.
// On context cancellation, the read completes in the background (no goroutine leak)
// because the next call waits for the mutex, which the background read holds until done.
func (w *worker) call(ctx context.Context, req any, resp any) error {
	w.mu.Lock()

	if err := w.stdin.Encode(req); err != nil {
		w.mu.Unlock()
		return fmt.Errorf("worker send: %w", err)
	}

	// Read response in a goroutine so we can respect ctx.
	// The goroutine holds the mutex until the read completes,
	// preventing concurrent scanner access from a subsequent call().
	type readResult struct {
		data []byte
		err  error
	}
	readCh := make(chan readResult, 1)
	go func() {
		defer w.mu.Unlock() // release mutex when read completes (or subprocess dies)
		if !w.stdout.Scan() {
			if err := w.stdout.Err(); err != nil {
				readCh <- readResult{nil, fmt.Errorf("worker recv: %w", err)}
			} else {
				readCh <- readResult{nil, fmt.Errorf("worker: subprocess closed stdout")}
			}
			return
		}
		// Copy bytes — scanner buffer is reused on next Scan()
		data := make([]byte, len(w.stdout.Bytes()))
		copy(data, w.stdout.Bytes())
		readCh <- readResult{data, nil}
	}()

	select {
	case res := <-readCh:
		if res.err != nil {
			return res.err
		}
		if err := json.Unmarshal(res.data, resp); err != nil {
			return fmt.Errorf("worker parse: %w (raw: %s)", err, string(res.data))
		}
		return nil
	case <-ctx.Done():
		// The goroutine still holds the mutex and will complete the read.
		// The next call() will block on mu.Lock() until the read finishes,
		// consuming the stale response — no scanner race, no goroutine leak.
		return ctx.Err()
	}
}

// alive checks if the subprocess is still running.
func (w *worker) alive() bool {
	if w == nil || w.cmd == nil || w.cmd.Process == nil {
		return false
	}
	// cmd.ProcessState is set after Wait() completes. If nil, process is still running.
	return w.cmd.ProcessState == nil
}

// stop gracefully shuts down the worker.
func (w *worker) stop() {
	if w == nil || w.cmd == nil || w.cmd.Process == nil {
		return
	}
	w.cmd.Process.Signal(os.Interrupt)
	done := make(chan struct{})
	go func() { w.cmd.Wait(); close(done) }()
	select {
	case <-done:
	case <-time.After(3 * time.Second):
		w.cmd.Process.Kill()
	}
}
