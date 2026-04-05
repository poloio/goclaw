package voice

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
)

// HTTPSynthesizer proxies TTS to any HTTP endpoint that accepts JSON {"text": "..."} and returns WAV.
// Compatible with our api.py /speak, OpenAI TTS, or any similar service.
type HTTPSynthesizer struct {
	URL        string // e.g. "http://localhost:8082/speak"
	APIKey     string // optional Bearer token
	sampleRate int
}

func NewHTTPSynthesizer(url, apiKey string, sampleRate int) *HTTPSynthesizer {
	if sampleRate == 0 {
		sampleRate = 22050
	}
	return &HTTPSynthesizer{URL: url, APIKey: apiKey, sampleRate: sampleRate}
}

func (h *HTTPSynthesizer) Synthesize(ctx context.Context, text string) ([]byte, error) {
	payload, _ := json.Marshal(map[string]string{"text": text})
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, h.URL, bytes.NewReader(payload))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	if h.APIKey != "" {
		req.Header.Set("Authorization", "Bearer "+h.APIKey)
	}

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("HTTP TTS request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("HTTP TTS status %d", resp.StatusCode)
	}

	return io.ReadAll(resp.Body)
}

func (h *HTTPSynthesizer) SampleRate() int {
	return h.sampleRate
}
