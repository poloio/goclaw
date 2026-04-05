package voice

import (
	"embed"
	"fmt"
	"os"
	"path/filepath"
)

//go:embed scripts/*.py
var embeddedScripts embed.FS

// scriptDir extracts embedded Python scripts to a temp directory on first call.
// Returns the directory path. Scripts persist for the process lifetime.
var scriptDir string

func getScriptDir() (string, error) {
	if scriptDir != "" {
		return scriptDir, nil
	}

	dir, err := os.MkdirTemp("", "goclaw-voice-scripts-*")
	if err != nil {
		return "", fmt.Errorf("create script dir: %w", err)
	}

	entries, err := embeddedScripts.ReadDir("scripts")
	if err != nil {
		return "", fmt.Errorf("read embedded scripts: %w", err)
	}

	for _, entry := range entries {
		data, err := embeddedScripts.ReadFile("scripts/" + entry.Name())
		if err != nil {
			return "", fmt.Errorf("read script %s: %w", entry.Name(), err)
		}
		path := filepath.Join(dir, entry.Name())
		if err := os.WriteFile(path, data, 0755); err != nil {
			return "", fmt.Errorf("write script %s: %w", entry.Name(), err)
		}
	}

	scriptDir = dir
	return dir, nil
}

func getScript(name string) (string, error) {
	dir, err := getScriptDir()
	if err != nil {
		return "", err
	}
	return filepath.Join(dir, name), nil
}
