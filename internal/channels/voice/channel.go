// Package voice provides a voice channel for GoClaw.
// It exposes an HTTP API that accepts audio (WAV), runs STT,
// routes text through the agent bus, runs TTS on the response,
// and returns audio back to the caller.
//
// This enables a remote client (e.g., a PC with a gaming headset
// or in-car mic) to interact with GoClaw agents via voice.
package voice

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"net/url"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"github.com/nextlevelbuilder/goclaw/internal/bus"
	"github.com/nextlevelbuilder/goclaw/internal/channels"
	"github.com/nextlevelbuilder/goclaw/internal/config"
)

const TypeVoice = "voice"

// Compile-time interface checks.
var _ channels.Channel = (*Channel)(nil)
var _ channels.WebhookChannel = (*Channel)(nil)

// Channel implements a voice I/O channel for GoClaw.
type Channel struct {
	*channels.BaseChannel
	cfg config.VoiceConfig

	// STT state
	sttCmd   string // path to STT binary or "whisper-ctranslate2"
	sttModel string

	// TTS state
	ttsVoicePath  string
	ttsConfigPath string
	ttsSampleRate string

	// Pending responses: chatID -> response channel
	pending   map[string]chan string
	pendingMu sync.Mutex
}

// New creates a new voice channel.
func New(cfg config.VoiceConfig, msgBus *bus.MessageBus) (*Channel, error) {
	base := channels.NewBaseChannel("voice", msgBus, cfg.AllowFrom)

	ch := &Channel{
		BaseChannel: base,
		cfg:         cfg,
		sttModel:    cfg.STTModel,
		pending:     make(map[string]chan string),
	}

	// Find TTS voice
	if err := ch.findVoice(); err != nil {
		slog.Warn("voice channel: no TTS voice found", "error", err)
	}

	return ch, nil
}

// findVoice locates the Piper TTS voice on disk.
func (c *Channel) findVoice() error {
	voicesDir := filepath.Join(os.Getenv("HOME"), ".local", "share", "piper", "voices")
	candidates := []string{"es_ES-davefx-medium", "es_MX-claude-high"}
	if c.cfg.TTSVoice != "" {
		candidates = append([]string{c.cfg.TTSVoice}, candidates...)
	}

	for _, name := range candidates {
		onnx := filepath.Join(voicesDir, name+".onnx")
		cfg := filepath.Join(voicesDir, name+".onnx.json")
		if _, err := os.Stat(onnx); err == nil {
			if _, err := os.Stat(cfg); err == nil {
				c.ttsVoicePath = onnx
				c.ttsConfigPath = cfg
				c.ttsSampleRate = "22050"
				// Try to read sample rate from config
				if data, err := os.ReadFile(cfg); err == nil {
					var parsed struct {
						Audio struct {
							SampleRate int `json:"sample_rate"`
						} `json:"audio"`
					}
					if json.Unmarshal(data, &parsed) == nil && parsed.Audio.SampleRate > 0 {
						c.ttsSampleRate = fmt.Sprintf("%d", parsed.Audio.SampleRate)
					}
				}
				slog.Info("voice channel: TTS voice found", "voice", name, "sample_rate", c.ttsSampleRate)
				return nil
			}
		}
	}
	return fmt.Errorf("no piper voice found in %s", voicesDir)
}

// Start is a no-op — the voice channel uses WebhookHandler to mount on the gateway mux.
func (c *Channel) Start(ctx context.Context) error {
	c.SetRunning(true)
	c.MarkHealthy("Listening")
	slog.Info("voice channel started", "port", "shared with gateway")
	return nil
}

// Stop gracefully shuts down the voice channel.
func (c *Channel) Stop(ctx context.Context) error {
	c.SetRunning(false)
	c.MarkStopped("")
	return nil
}

// Send receives an outbound message from the agent and delivers it to the
// waiting HTTP request via the pending response channel.
func (c *Channel) Send(ctx context.Context, msg bus.OutboundMessage) error {
	c.pendingMu.Lock()
	ch, ok := c.pending[msg.ChatID]
	c.pendingMu.Unlock()

	if !ok {
		slog.Warn("voice: no pending request for chatID", "chatID", msg.ChatID)
		return nil
	}

	select {
	case ch <- msg.Content:
	case <-ctx.Done():
		return ctx.Err()
	case <-time.After(30 * time.Second):
		return fmt.Errorf("voice: timeout delivering response to chatID %s", msg.ChatID)
	}
	return nil
}

// IsAllowed checks the allowlist. For voice, we default to open (single-user car).
func (c *Channel) IsAllowed(senderID string) bool {
	if len(c.cfg.AllowFrom) == 0 {
		return true // open by default for single-user deployments
	}
	return c.BaseChannel.IsAllowed(senderID)
}

// WebhookHandler returns the HTTP handler mounted on the gateway mux.
func (c *Channel) WebhookHandler() (string, http.Handler) {
	mux := http.NewServeMux()
	mux.HandleFunc("/voice/health", c.handleHealth)
	mux.HandleFunc("/voice/pipeline", c.handlePipeline)
	mux.HandleFunc("/voice/chat", c.handleChat)
	mux.HandleFunc("/voice/speak", c.handleSpeak)
	return "/voice/", mux
}

// --- HTTP Handlers ---

func (c *Channel) handleHealth(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]string{
		"status":    "ok",
		"tts_voice": c.ttsVoicePath,
		"stt_model": c.sttModel,
	})
}

// handlePipeline: POST audio WAV -> STT -> agent -> TTS -> return WAV
func (c *Channel) handlePipeline(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "POST only", http.StatusMethodNotAllowed)
		return
	}

	// Read uploaded audio
	if err := r.ParseMultipartForm(10 << 20); err != nil { // 10MB max
		http.Error(w, "bad request", http.StatusBadRequest)
		return
	}
	file, _, err := r.FormFile("audio")
	if err != nil {
		http.Error(w, "no audio file", http.StatusBadRequest)
		return
	}
	defer file.Close()

	// Save to temp file for STT
	tmpWav, err := os.CreateTemp("", "voice-*.wav")
	if err != nil {
		http.Error(w, "internal error", http.StatusInternalServerError)
		return
	}
	defer os.Remove(tmpWav.Name())
	io.Copy(tmpWav, file)
	tmpWav.Close()

	// STT
	text, err := c.transcribe(tmpWav.Name())
	if err != nil {
		slog.Error("voice: STT failed", "error", err)
		http.Error(w, "STT failed", http.StatusInternalServerError)
		return
	}
	if strings.TrimSpace(text) == "" {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]string{"error": "no speech detected"})
		return
	}

	// Create a unique chat ID for this request and register a response channel
	chatID := fmt.Sprintf("voice-%d", time.Now().UnixNano())
	respCh := make(chan string, 1)
	c.pendingMu.Lock()
	c.pending[chatID] = respCh
	c.pendingMu.Unlock()
	defer func() {
		c.pendingMu.Lock()
		delete(c.pending, chatID)
		c.pendingMu.Unlock()
	}()

	// Publish to bus — this triggers the agent loop
	senderID := r.Header.Get("X-Voice-Sender")
	if senderID == "" {
		senderID = "driver"
	}
	c.BaseChannel.HandleMessage(senderID, chatID, text, nil, nil, "direct")

	// Wait for agent response
	var reply string
	select {
	case reply = <-respCh:
	case <-time.After(30 * time.Second):
		http.Error(w, "agent timeout", http.StatusGatewayTimeout)
		return
	}

	// TTS
	wavBytes, err := c.synthesize(reply)
	if err != nil {
		slog.Error("voice: TTS failed", "error", err)
		// Return text even if TTS fails
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]string{
			"text":  text,
			"reply": reply,
			"error": "TTS failed",
		})
		return
	}

	// Return audio with text in headers
	w.Header().Set("Content-Type", "audio/wav")
	w.Header().Set("X-User-Text", url.QueryEscape(text))
	w.Header().Set("X-Reply-Text", url.QueryEscape(reply))
	w.Write(wavBytes)
}

// handleChat: POST JSON {"message": "..."} -> agent -> JSON {"reply": "..."}
func (c *Channel) handleChat(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "POST only", http.StatusMethodNotAllowed)
		return
	}

	var req struct {
		Message string `json:"message"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil || req.Message == "" {
		http.Error(w, "bad request", http.StatusBadRequest)
		return
	}

	chatID := fmt.Sprintf("voice-%d", time.Now().UnixNano())
	respCh := make(chan string, 1)
	c.pendingMu.Lock()
	c.pending[chatID] = respCh
	c.pendingMu.Unlock()
	defer func() {
		c.pendingMu.Lock()
		delete(c.pending, chatID)
		c.pendingMu.Unlock()
	}()

	senderID := "driver"
	c.BaseChannel.HandleMessage(senderID, chatID, req.Message, nil, nil, "direct")

	var reply string
	select {
	case reply = <-respCh:
	case <-time.After(30 * time.Second):
		http.Error(w, "agent timeout", http.StatusGatewayTimeout)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]string{"reply": reply})
}

// handleSpeak: POST JSON {"text": "..."} -> TTS -> return WAV
func (c *Channel) handleSpeak(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "POST only", http.StatusMethodNotAllowed)
		return
	}

	var req struct {
		Text string `json:"text"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil || req.Text == "" {
		http.Error(w, "bad request", http.StatusBadRequest)
		return
	}

	wavBytes, err := c.synthesize(req.Text)
	if err != nil {
		http.Error(w, "TTS failed", http.StatusInternalServerError)
		return
	}

	w.Header().Set("Content-Type", "audio/wav")
	w.Write(wavBytes)
}

// --- STT ---

// transcribe runs faster-whisper on a WAV file and returns the text.
func (c *Channel) transcribe(wavPath string) (string, error) {
	// Use a small Python wrapper since faster-whisper is Python-only
	script := fmt.Sprintf(`
import sys, json
from faster_whisper import WhisperModel
model = WhisperModel("%s", device="cpu", compute_type="int8")
segments, info = model.transcribe(sys.argv[1], language="es", beam_size=1, vad_filter=True)
text = " ".join(s.text.strip() for s in segments)
print(json.dumps({"text": text.strip()}))
`, c.sttModel)

	cmd := exec.Command("python3", "-c", script, wavPath)
	// Use the jarvis venv if available
	venvPython := filepath.Join(os.Getenv("HOME"), "jarvis-env", "bin", "python3")
	if _, err := os.Stat(venvPython); err == nil {
		cmd = exec.Command(venvPython, "-c", script, wavPath)
	}

	out, err := cmd.Output()
	if err != nil {
		if exitErr, ok := err.(*exec.ExitError); ok {
			return "", fmt.Errorf("STT error: %s", string(exitErr.Stderr))
		}
		return "", err
	}

	var result struct {
		Text string `json:"text"`
	}
	if err := json.Unmarshal(out, &result); err != nil {
		return "", fmt.Errorf("STT parse error: %w", err)
	}
	return result.Text, nil
}

// --- TTS ---

// synthesize runs Piper TTS on text and returns WAV bytes.
func (c *Channel) synthesize(text string) ([]byte, error) {
	if c.ttsVoicePath == "" {
		return nil, fmt.Errorf("no TTS voice configured")
	}

	tmpWav, err := os.CreateTemp("", "voice-tts-*.wav")
	if err != nil {
		return nil, err
	}
	tmpPath := tmpWav.Name()
	tmpWav.Close()
	defer os.Remove(tmpPath)

	cmd := exec.Command("piper",
		"--model", c.ttsVoicePath,
		"--config", c.ttsConfigPath,
		"--output_file", tmpPath,
	)
	cmd.Stdin = strings.NewReader(text)

	if out, err := cmd.CombinedOutput(); err != nil {
		return nil, fmt.Errorf("piper error: %s: %w", string(out), err)
	}

	return os.ReadFile(tmpPath)
}
