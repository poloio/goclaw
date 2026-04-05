package voice

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"mime/multipart"
	"net/http"
	"os"
)

// HTTPTranscriber proxies STT to any HTTP endpoint that accepts WAV and returns JSON {"text": "..."}.
// Compatible with OpenAI Whisper API, our api.py /transcribe, or any similar service.
type HTTPTranscriber struct {
	URL      string // e.g. "http://localhost:8082/transcribe"
	APIKey   string // optional Bearer token
	Language string
}

func (h *HTTPTranscriber) Transcribe(ctx context.Context, wavPath string) (string, error) {
	f, err := os.Open(wavPath)
	if err != nil {
		return "", err
	}
	defer f.Close()

	var body bytes.Buffer
	writer := multipart.NewWriter(&body)
	part, err := writer.CreateFormFile("audio", "recording.wav")
	if err != nil {
		return "", err
	}
	io.Copy(part, f)
	if h.Language != "" {
		writer.WriteField("language", h.Language)
	}
	writer.Close()

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, h.URL, &body)
	if err != nil {
		return "", err
	}
	req.Header.Set("Content-Type", writer.FormDataContentType())
	if h.APIKey != "" {
		req.Header.Set("Authorization", "Bearer "+h.APIKey)
	}

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return "", fmt.Errorf("HTTP STT request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("HTTP STT status %d", resp.StatusCode)
	}

	var result struct {
		Text string `json:"text"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return "", fmt.Errorf("HTTP STT parse: %w", err)
	}
	return result.Text, nil
}
