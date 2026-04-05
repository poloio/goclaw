// Package voice provides a voice I/O channel for GoClaw.
//
// Two modes of operation:
//   - Local: mic/speaker connected to the device (Jetson). Runs a continuous
//     listen loop: AudioSource → Transcriber → agent bus → Synthesizer → AudioSink.
//   - HTTP: remote client sends audio over HTTP, gets audio back. Endpoints
//     mounted on the gateway mux via WebhookChannel.
//
// All STT/TTS is behind interfaces (Transcriber, Synthesizer, AudioSource, AudioSink)
// so backends can be swapped without touching the channel logic.
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
	"strings"
	"sync"
	"time"

	"github.com/nextlevelbuilder/goclaw/internal/bus"
	"github.com/nextlevelbuilder/goclaw/internal/channels"
	"github.com/nextlevelbuilder/goclaw/internal/config"
)

// Compile-time interface checks.
var _ channels.Channel = (*Channel)(nil)
var _ channels.WebhookChannel = (*Channel)(nil)

// Channel implements a voice I/O channel for GoClaw.
type Channel struct {
	*channels.BaseChannel
	cfg config.VoiceConfig

	// Pluggable components
	stt  Transcriber
	tts  Synthesizer
	mic  AudioSource  // nil if local mode is disabled
	spk  AudioSink    // nil if local mode is disabled

	// Pending responses: chatID -> response channel (for HTTP and local modes)
	pending   map[string]chan string
	pendingMu sync.Mutex

	// Local loop control
	cancelLocal context.CancelFunc
}

// New creates a new voice channel with the configured STT/TTS backends.
func New(cfg config.VoiceConfig, msgBus *bus.MessageBus) (*Channel, error) {
	base := channels.NewBaseChannel("voice", msgBus, cfg.AllowFrom)

	ch := &Channel{
		BaseChannel: base,
		cfg:         cfg,
		pending:     make(map[string]chan string),
	}

	// Initialize STT
	ch.stt = NewWhisperTranscriber(cfg.STTModel, cfg.Language)

	// Initialize TTS
	tts, err := NewPiperSynthesizer(cfg.TTSVoice, nil)
	if err != nil {
		slog.Warn("voice: no TTS voice, audio output disabled", "error", err)
	} else {
		ch.tts = tts
	}

	// Initialize local audio if enabled (default: true)
	if !cfg.DisableLocal {
		ch.mic = NewLocalMic()
		ch.spk = NewLocalSpeaker(cfg.ALSADevice)
		slog.Info("voice: local audio enabled")
	}

	return ch, nil
}

// WithTranscriber replaces the STT backend.
func (c *Channel) WithTranscriber(t Transcriber) { c.stt = t }

// WithSynthesizer replaces the TTS backend.
func (c *Channel) WithSynthesizer(s Synthesizer) { c.tts = s }

// WithAudioSource replaces the mic/audio input.
func (c *Channel) WithAudioSource(src AudioSource) { c.mic = src }

// WithAudioSink replaces the speaker/audio output.
func (c *Channel) WithAudioSink(sink AudioSink) { c.spk = sink }

// Start begins the local voice loop (if enabled). HTTP mode is always available
// via WebhookHandler regardless.
func (c *Channel) Start(ctx context.Context) error {
	c.SetRunning(true)
	c.MarkHealthy("Listening")

	if c.mic != nil {
		loopCtx, cancel := context.WithCancel(ctx)
		c.cancelLocal = cancel
		go c.localLoop(loopCtx)
		slog.Info("voice: local listen loop started")
	}

	slog.Info("voice channel started")
	return nil
}

// Stop gracefully shuts down the voice channel.
func (c *Channel) Stop(ctx context.Context) error {
	if c.cancelLocal != nil {
		c.cancelLocal()
	}
	if c.mic != nil {
		c.mic.Close()
	}
	if c.spk != nil {
		c.spk.Close()
	}
	c.SetRunning(false)
	c.MarkStopped("")
	return nil
}

// Send receives an outbound message from the agent and delivers it to the
// waiting response channel (used by both local loop and HTTP handlers).
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

// IsAllowed defaults to open for single-user deployments.
func (c *Channel) IsAllowed(senderID string) bool {
	if len(c.cfg.AllowFrom) == 0 {
		return true
	}
	return c.BaseChannel.IsAllowed(senderID)
}

// --- Local voice loop ---

// localLoop continuously listens for speech, transcribes, sends to agent, synthesizes, and plays.
func (c *Channel) localLoop(ctx context.Context) {
	for {
		select {
		case <-ctx.Done():
			return
		default:
		}

		reply, err := c.processOneUtterance(ctx)
		if err != nil {
			slog.Error("voice: local loop error", "error", err)
			time.Sleep(500 * time.Millisecond)
			continue
		}
		if reply == "" {
			continue // no speech detected, loop again
		}
	}
}

// processOneUtterance handles one full listen→STT→agent→TTS→play cycle.
func (c *Channel) processOneUtterance(ctx context.Context) (string, error) {
	// 1. Listen for speech
	wavPath, err := c.mic.ListenOnce(ctx)
	if err != nil {
		return "", fmt.Errorf("mic: %w", err)
	}
	if wavPath == "" {
		return "", nil // no speech
	}
	defer os.Remove(wavPath)

	// 2. Transcribe
	text, err := c.stt.Transcribe(ctx, wavPath)
	if err != nil {
		return "", fmt.Errorf("STT: %w", err)
	}
	if strings.TrimSpace(text) == "" {
		return "", nil
	}
	slog.Info("voice: heard", "text", text)

	// 3. Send to agent and wait for response
	reply, err := c.sendAndWait(ctx, "driver", text)
	if err != nil {
		return "", fmt.Errorf("agent: %w", err)
	}
	slog.Info("voice: reply", "text", reply)

	// 4. Synthesize and play
	if c.tts != nil && c.spk != nil {
		wavData, err := c.tts.Synthesize(ctx, reply)
		if err != nil {
			slog.Error("voice: TTS failed", "error", err)
		} else if err := c.spk.Play(ctx, wavData); err != nil {
			slog.Error("voice: playback failed", "error", err)
		}
	}

	return reply, nil
}

// sendAndWait publishes a message to the bus and blocks until the agent responds.
func (c *Channel) sendAndWait(ctx context.Context, senderID, text string) (string, error) {
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

	c.BaseChannel.HandleMessage(senderID, chatID, text, nil, nil, "direct")

	select {
	case reply := <-respCh:
		return reply, nil
	case <-ctx.Done():
		return "", ctx.Err()
	case <-time.After(30 * time.Second):
		return "", fmt.Errorf("agent response timeout")
	}
}

// --- HTTP mode (WebhookChannel) ---

// WebhookHandler returns HTTP endpoints for remote clients.
func (c *Channel) WebhookHandler() (string, http.Handler) {
	mux := http.NewServeMux()
	mux.HandleFunc("/voice/health", c.handleHealth)
	mux.HandleFunc("/voice/pipeline", c.handlePipeline)
	mux.HandleFunc("/voice/chat", c.handleChat)
	mux.HandleFunc("/voice/speak", c.handleSpeak)
	return "/voice/", mux
}

func (c *Channel) handleHealth(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	info := map[string]any{
		"status":     "ok",
		"local_mode": c.mic != nil,
		"has_tts":    c.tts != nil,
	}
	json.NewEncoder(w).Encode(info)
}

// handlePipeline: POST WAV → STT → agent → TTS → return WAV
func (c *Channel) handlePipeline(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "POST only", http.StatusMethodNotAllowed)
		return
	}
	if err := r.ParseMultipartForm(10 << 20); err != nil {
		http.Error(w, "bad request", http.StatusBadRequest)
		return
	}
	file, _, err := r.FormFile("audio")
	if err != nil {
		http.Error(w, "no audio file", http.StatusBadRequest)
		return
	}
	defer file.Close()

	tmp, err := os.CreateTemp("", "voice-http-*.wav")
	if err != nil {
		http.Error(w, "internal error", http.StatusInternalServerError)
		return
	}
	defer os.Remove(tmp.Name())
	io.Copy(tmp, file)
	tmp.Close()

	ctx := r.Context()
	text, err := c.stt.Transcribe(ctx, tmp.Name())
	if err != nil {
		http.Error(w, "STT failed", http.StatusInternalServerError)
		return
	}
	if strings.TrimSpace(text) == "" {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]string{"error": "no speech detected"})
		return
	}

	senderID := r.Header.Get("X-Voice-Sender")
	if senderID == "" {
		senderID = "driver"
	}
	reply, err := c.sendAndWait(ctx, senderID, text)
	if err != nil {
		http.Error(w, "agent timeout", http.StatusGatewayTimeout)
		return
	}

	if c.tts == nil {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]string{"text": text, "reply": reply})
		return
	}

	wavBytes, err := c.tts.Synthesize(ctx, reply)
	if err != nil {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]string{"text": text, "reply": reply, "error": "TTS failed"})
		return
	}

	w.Header().Set("Content-Type", "audio/wav")
	w.Header().Set("X-User-Text", url.QueryEscape(text))
	w.Header().Set("X-Reply-Text", url.QueryEscape(reply))
	w.Write(wavBytes)
}

// handleChat: POST JSON {"message":"..."} → agent → JSON {"reply":"..."}
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

	reply, err := c.sendAndWait(r.Context(), "driver", req.Message)
	if err != nil {
		http.Error(w, "agent timeout", http.StatusGatewayTimeout)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]string{"reply": reply})
}

// handleSpeak: POST JSON {"text":"..."} → TTS → WAV
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
	if c.tts == nil {
		http.Error(w, "no TTS configured", http.StatusServiceUnavailable)
		return
	}

	wavBytes, err := c.tts.Synthesize(r.Context(), req.Text)
	if err != nil {
		http.Error(w, "TTS failed", http.StatusInternalServerError)
		return
	}

	w.Header().Set("Content-Type", "audio/wav")
	w.Write(wavBytes)
}
