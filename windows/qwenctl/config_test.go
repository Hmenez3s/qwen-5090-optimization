package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestConfigDefaultsAndProfiles(t *testing.T) {
	defaults := DefaultConfig()
	if err := defaults.Validate(); err != nil {
		t.Fatalf("defaults rejected: %v", err)
	}
	if got := defaults.Profile(); got != "VALIDATED" {
		t.Fatalf("profile = %q", got)
	}
	custom := defaults
	custom.RandomSeed = 17
	if err := custom.Validate(); err != nil {
		t.Fatalf("custom rejected: %v", err)
	}
	if got := custom.Profile(); got != "CUSTOM_UNVALIDATED" {
		t.Fatalf("custom profile = %q", got)
	}
}

func TestConfigRangesAndCrossField(t *testing.T) {
	cases := []Config{
		{ContextLength: 131071, MaxTotalTokens: 131328, RandomSeed: 7319, StartupTimeout: 900},
		{ContextLength: 131200, MaxTotalTokens: 131199, RandomSeed: 7319, StartupTimeout: 900},
		{ContextLength: 131200, MaxTotalTokens: 131328, RandomSeed: -1, StartupTimeout: 900},
		{ContextLength: 131200, MaxTotalTokens: 131328, RandomSeed: 7319, StartupTimeout: 59},
	}
	for index, candidate := range cases {
		if err := candidate.Validate(); err == nil {
			t.Fatalf("case %d unexpectedly valid", index)
		}
	}
}

func TestInvalidConfigSetDoesNotWrite(t *testing.T) {
	dir := t.TempDir()
	if err := writeConfig(dir, DefaultConfig()); err != nil {
		t.Fatal(err)
	}
	before, err := os.ReadFile(configPath(dir))
	if err != nil {
		t.Fatal(err)
	}
	if _, err := setConfigValue(DefaultConfig(), "max-total-tokens", "131199"); err == nil {
		t.Fatal("cross-field update should fail while max is below context")
	}
	after, err := os.ReadFile(configPath(dir))
	if err != nil {
		t.Fatal(err)
	}
	if string(before) != string(after) {
		t.Fatal("invalid update changed config bytes")
	}
}

func TestConfigRoundTripAndNoUnknownKeys(t *testing.T) {
	dir := t.TempDir()
	custom := DefaultConfig()
	custom.ContextLength = 131072
	custom.MaxTotalTokens = 131072
	custom.RandomSeed = 0
	custom.StartupTimeout = 1800
	if err := writeConfig(dir, custom); err != nil {
		t.Fatal(err)
	}
	got, err := loadConfig(dir)
	if err != nil {
		t.Fatal(err)
	}
	if got != custom {
		t.Fatalf("round trip = %#v, want %#v", got, custom)
	}
	contents, err := os.ReadFile(configPath(dir))
	if err != nil {
		t.Fatal(err)
	}
	var raw map[string]any
	if err := json.Unmarshal(contents, &raw); err != nil {
		t.Fatal(err)
	}
	if len(raw) != 4 {
		t.Fatalf("config has %d keys, want 4", len(raw))
	}
	if err := os.WriteFile(configPath(dir), []byte(`{"context-length":131200} {}`), 0600); err != nil {
		t.Fatal(err)
	}
	if _, err := loadConfig(dir); err == nil {
		t.Fatal("trailing JSON accepted")
	}
}

func TestConfigCommandUsesTestHomeAndShowsPendingProfile(t *testing.T) {
	dir := t.TempDir()
	old := os.Getenv("QWENCTL_HOME")
	defer os.Setenv("QWENCTL_HOME", old)
	if err := os.Setenv("QWENCTL_HOME", dir); err != nil {
		t.Fatal(err)
	}
	result := runCommand(NewApp(), []string{"config", "show", "--json"})
	if !result.response.OK || result.response.Code != "CONFIG" {
		t.Fatalf("show response: %#v", result.response)
	}
	if result.response.Profile != "VALIDATED" {
		t.Fatalf("show profile = %s", result.response.Profile)
	}
	set := runCommand(NewApp(), []string{"config", "set", "random-seed", "17", "--json"})
	if !set.response.OK || set.response.Profile != "CUSTOM_UNVALIDATED" {
		t.Fatalf("set response: %#v", set.response)
	}
	if _, err := os.Stat(filepath.Join(dir, "config.json")); err != nil {
		t.Fatal(err)
	}
	bad := runCommand(NewApp(), []string{"config", "set", "context-length", "131201", "--json"})
	if bad.response.OK || bad.response.Code != "INVALID_CONFIG" {
		t.Fatalf("bad response: %#v", bad.response)
	}
	if strings.Contains(string(bad.response.Data), "131201") {
		t.Fatal("invalid value appeared in response data")
	}
}
