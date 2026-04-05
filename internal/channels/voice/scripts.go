package voice

import (
	"embed"
	"fmt"
	"os"
	"path/filepath"
	"sync"
)

//go:embed scripts/*.py
var embeddedScripts embed.FS

var (
	scriptDirOnce sync.Once
	scriptDirPath string
	scriptDirErr  error
)

func getScriptDir() (string, error) {
	scriptDirOnce.Do(func() {
		dir, err := os.MkdirTemp("", "goclaw-voice-scripts-*")
		if err != nil {
			scriptDirErr = fmt.Errorf("create script dir: %w", err)
			return
		}

		entries, err := embeddedScripts.ReadDir("scripts")
		if err != nil {
			os.RemoveAll(dir)
			scriptDirErr = fmt.Errorf("read embedded scripts: %w", err)
			return
		}

		for _, entry := range entries {
			data, err := embeddedScripts.ReadFile("scripts/" + entry.Name())
			if err != nil {
				os.RemoveAll(dir)
				scriptDirErr = fmt.Errorf("read script %s: %w", entry.Name(), err)
				return
			}
			path := filepath.Join(dir, entry.Name())
			if err := os.WriteFile(path, data, 0755); err != nil {
				os.RemoveAll(dir)
				scriptDirErr = fmt.Errorf("write script %s: %w", entry.Name(), err)
				return
			}
		}

		scriptDirPath = dir
	})
	return scriptDirPath, scriptDirErr
}

func getScript(name string) (string, error) {
	dir, err := getScriptDir()
	if err != nil {
		return "", err
	}
	return filepath.Join(dir, name), nil
}

// CleanupScripts removes the temporary script directory. Call on process shutdown.
func CleanupScripts() {
	if scriptDirPath != "" {
		os.RemoveAll(scriptDirPath)
	}
}
