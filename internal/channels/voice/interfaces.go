package voice

import "context"

// Transcriber converts audio to text (STT).
// Implementations: WhisperTranscriber (subprocess), HTTPTranscriber (proxy to any API).
type Transcriber interface {
	// Transcribe takes a path to a WAV file and returns the transcribed text.
	Transcribe(ctx context.Context, wavPath string) (string, error)
}

// Synthesizer converts text to audio (TTS).
// Implementations: PiperSynthesizer (subprocess), HTTPSynthesizer (proxy to any API).
type Synthesizer interface {
	// Synthesize takes text and returns WAV audio bytes.
	Synthesize(ctx context.Context, text string) ([]byte, error)
	// SampleRate returns the output audio sample rate in Hz.
	SampleRate() int
}

// AudioSource provides audio input.
// Implementations: LocalMic (ALSA capture), HTTPSource (uploaded WAV).
type AudioSource interface {
	// ListenOnce blocks until an utterance is captured (via VAD), then returns the WAV path.
	// The caller is responsible for deleting the file after use.
	ListenOnce(ctx context.Context) (wavPath string, err error)
	// Close releases audio resources.
	Close() error
}

// AudioSink plays audio output.
// Implementations: LocalSpeaker (ALSA playback), HTTPSink (return bytes in HTTP response).
type AudioSink interface {
	// Play takes WAV audio bytes and plays them.
	Play(ctx context.Context, wavData []byte) error
	// Close releases audio resources.
	Close() error
}
