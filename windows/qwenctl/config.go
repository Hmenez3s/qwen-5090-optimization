package main

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
)

const (
	defaultContextLength  int64 = 131200
	defaultMaxTotalTokens int64 = 131328
	defaultRandomSeed     int64 = 7319
	defaultStartupTimeout int64 = 900
	minimumContextLength  int64 = 131072
	maximumContextLength  int64 = 131200
	maximumTotalTokens    int64 = 131328
	maximumRandomSeed     int64 = 2147483647
	minimumStartupTimeout int64 = 60
	maximumStartupTimeout int64 = 1800
)

type Config struct {
	ContextLength  int64 `json:"context-length"`
	MaxTotalTokens int64 `json:"max-total-tokens"`
	RandomSeed     int64 `json:"random-seed"`
	StartupTimeout int64 `json:"startup-timeout-seconds"`
}

func DefaultConfig() Config {
	return Config{
		ContextLength:  defaultContextLength,
		MaxTotalTokens: defaultMaxTotalTokens,
		RandomSeed:     defaultRandomSeed,
		StartupTimeout: defaultStartupTimeout,
	}
}

func (c Config) Validate() error {
	if c.ContextLength < minimumContextLength || c.ContextLength > maximumContextLength {
		return fmt.Errorf("context-length deve estar entre %d e %d", minimumContextLength, maximumContextLength)
	}
	if c.MaxTotalTokens < c.ContextLength || c.MaxTotalTokens > maximumTotalTokens {
		return fmt.Errorf("max-total-tokens deve ser >= context-length e <= %d", maximumTotalTokens)
	}
	if c.RandomSeed < 0 || c.RandomSeed > maximumRandomSeed {
		return fmt.Errorf("random-seed deve estar entre 0 e %d", maximumRandomSeed)
	}
	if c.StartupTimeout < minimumStartupTimeout || c.StartupTimeout > maximumStartupTimeout {
		return fmt.Errorf("startup-timeout-seconds deve estar entre %d e %d", minimumStartupTimeout, maximumStartupTimeout)
	}
	return nil
}

func (c Config) Profile() string {
	if c == DefaultConfig() {
		return "VALIDATED"
	}
	return "CUSTOM_UNVALIDATED"
}

func configMap(c Config) map[string]int64 {
	return map[string]int64{
		"context-length":          c.ContextLength,
		"max-total-tokens":        c.MaxTotalTokens,
		"random-seed":             c.RandomSeed,
		"startup-timeout-seconds": c.StartupTimeout,
	}
}

func configView(c Config) map[string]any {
	return map[string]any{
		"context-length":          c.ContextLength,
		"max-total-tokens":        c.MaxTotalTokens,
		"random-seed":             c.RandomSeed,
		"startup-timeout-seconds": c.StartupTimeout,
		"profile":                 c.Profile(),
	}
}

func configDir() (string, error) {
	if override := strings.TrimSpace(os.Getenv("QWENCTL_HOME")); override != "" {
		return filepath.Clean(override), nil
	}
	if runtime.GOOS == "windows" {
		if local := os.Getenv("LOCALAPPDATA"); local != "" {
			return filepath.Join(local, "QwenCtl"), nil
		}
		return "", errors.New("%LOCALAPPDATA% não está definido")
	}
	base, err := os.UserConfigDir()
	if err != nil {
		return "", fmt.Errorf("não foi possível localizar configuração local: %w", err)
	}
	return filepath.Join(base, "QwenCtl"), nil
}

func configPath(dir string) string {
	return filepath.Join(dir, "config.json")
}

func statePath(dir string) string {
	return filepath.Join(dir, "state.json")
}

func loadConfig(dir string) (Config, error) {
	path := configPath(dir)
	if info, statErr := os.Lstat(path); statErr == nil && info.Mode()&os.ModeSymlink != 0 {
		return Config{}, errors.New("config.json symlink não é aceito")
	}
	contents, err := os.ReadFile(path)
	if errors.Is(err, os.ErrNotExist) {
		return DefaultConfig(), nil
	}
	if err != nil {
		return Config{}, fmt.Errorf("não foi possível ler %s: %w", path, err)
	}
	var raw map[string]json.RawMessage
	decoder := json.NewDecoder(strings.NewReader(string(contents)))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&raw); err != nil {
		return Config{}, fmt.Errorf("config.json inválido: %w", err)
	}
	var trailing any
	if err := decoder.Decode(&trailing); err == nil {
		return Config{}, errors.New("config.json contém dados após o objeto")
	} else if !errors.Is(err, io.EOF) {
		return Config{}, fmt.Errorf("config.json contém JSON inválido após o objeto: %w", err)
	}
	allowed := map[string]bool{
		"context-length":          true,
		"max-total-tokens":        true,
		"random-seed":             true,
		"startup-timeout-seconds": true,
	}
	for key := range raw {
		if !allowed[key] {
			return Config{}, fmt.Errorf("chave de configuração não permitida: %s", key)
		}
	}
	result := DefaultConfig()
	for key, value := range raw {
		var parsed int64
		if err := json.Unmarshal(value, &parsed); err != nil {
			return Config{}, fmt.Errorf("%s deve ser inteiro: %w", key, err)
		}
		switch key {
		case "context-length":
			result.ContextLength = parsed
		case "max-total-tokens":
			result.MaxTotalTokens = parsed
		case "random-seed":
			result.RandomSeed = parsed
		case "startup-timeout-seconds":
			result.StartupTimeout = parsed
		}
	}
	if err := result.Validate(); err != nil {
		return Config{}, fmt.Errorf("config.json inválido: %w", err)
	}
	return result, nil
}

func writeAtomicJSON(path string, value any) error {
	if info, err := os.Lstat(path); err == nil && info.Mode()&os.ModeSymlink != 0 {
		return errors.New("destino JSON symlink não é aceito")
	}
	parent := filepath.Dir(path)
	if err := os.MkdirAll(parent, 0700); err != nil {
		return err
	}
	temporary, err := os.CreateTemp(parent, ".qwenctl-atomic-*")
	if err != nil {
		return err
	}
	temporaryName := temporary.Name()
	keep := false
	defer func() {
		_ = temporary.Close()
		if !keep {
			_ = os.Remove(temporaryName)
		}
	}()
	if err := temporary.Chmod(0600); err != nil {
		return err
	}
	encoder := json.NewEncoder(temporary)
	encoder.SetIndent("", "  ")
	if err := encoder.Encode(value); err != nil {
		return err
	}
	if err := temporary.Sync(); err != nil {
		return err
	}
	if err := temporary.Close(); err != nil {
		return err
	}
	if err := os.Rename(temporaryName, path); err != nil {
		return fmt.Errorf("substituição atômica falhou: %w", err)
	}
	keep = true
	return nil
}

func writeConfig(dir string, c Config) error {
	if err := c.Validate(); err != nil {
		return err
	}
	return writeAtomicJSON(configPath(dir), configMap(c))
}

func parseConfigKey(key string) bool {
	switch key {
	case "context-length", "max-total-tokens", "random-seed", "startup-timeout-seconds":
		return true
	default:
		return false
	}
}

func setConfigValue(c Config, key string, raw string) (Config, error) {
	if !parseConfigKey(key) {
		return Config{}, fmt.Errorf("chave não permitida: %s", key)
	}
	value, err := strconv.ParseInt(raw, 10, 64)
	if err != nil {
		return Config{}, fmt.Errorf("%s deve ser inteiro", key)
	}
	switch key {
	case "context-length":
		c.ContextLength = value
	case "max-total-tokens":
		c.MaxTotalTokens = value
	case "random-seed":
		c.RandomSeed = value
	case "startup-timeout-seconds":
		c.StartupTimeout = value
	}
	if err := c.Validate(); err != nil {
		return Config{}, err
	}
	return c, nil
}
