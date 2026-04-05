package voice

import "context"

// Transcriber converts audio to text (STT).
// Implementations: WhisperTranscriber (persistent worker), HTTPTranscriber (proxy).
type Transcriber interface {
	Transcribe(ctx context.Context, wavPath string) (string, error)
}

// Synthesizer converts text to audio (TTS).
// Implementations: PiperSynthesizer (subprocess), HTTPSynthesizer (proxy).
type Synthesizer interface {
	Synthesize(ctx context.Context, text string) ([]byte, error)
	SampleRate() int
}

// AudioSource provides audio input.
// Implementations: LocalMic (Silero VAD worker), WakeWordMic (openWakeWord worker).
type AudioSource interface {
	// ListenOnce blocks until an utterance is captured, returns path to WAV file.
	// Returns ("", nil) if no speech was detected (not an error).
	// Caller must delete the file after use.
	ListenOnce(ctx context.Context) (wavPath string, err error)
	Close() error
}

// AudioSink plays audio output.
// Implementations: LocalSpeaker (aplay).
type AudioSink interface {
	Play(ctx context.Context, wavData []byte) error
	Close() error
}

// Startable is optionally implemented by components that need async initialization
// (e.g., launching a persistent worker subprocess).
type Startable interface {
	Start(ctx context.Context) error
}
