package main

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func responseJSON(code string, ok bool, profile string, data map[string]any) []byte {
	value := successEnvelope(code, profile, data)
	value.OK = ok
	if !ok {
		value.Error = &ManagerError{Kind: "test", Message: "fixture refusal"}
	}
	encoded, _ := json.Marshal(value)
	return encoded
}

func TestControllerCommandIsFixedAndRootBound(t *testing.T) {
	root := linuxRunBase + "/qwenctl-20260910T010203Z-1234abcd"
	command, err := buildControllerCommand(root, root+"/launch.json", `C:\logs\out.log`, `C:\logs\err.log`)
	if err != nil {
		t.Fatal(err)
	}
	for _, part := range []string{
		`C:\Windows\System32\cmd.exe /d /c`,
		`C:\Windows\System32\wsl.exe -d Ubuntu -- python3`,
		root + "/exec-manifest.py",
		root + "/launch.json",
		`1>"C:\logs\out.log" 2>"C:\logs\err.log"`,
	} {
		if !strings.Contains(command, part) {
			t.Fatalf("command %q does not contain %q", command, part)
		}
	}
	if _, err := buildControllerCommand(root, root+"/other.json", `C:\logs\out.log`, `C:\logs\err.log`); err == nil {
		t.Fatal("unrelated launch path accepted")
	}
}

func TestEmbeddedProgramContainsNoTemporaryManagerPath(t *testing.T) {
	program := embeddedPythonProgram([]byte(`{"schema_version":1,"operation":"probe","config":{}}`))
	text := string(program)
	if !strings.Contains(text, "wsl_manager.py") || !strings.Contains(text, "run_protocol") {
		t.Fatal("embedded protocol wrapper incomplete")
	}
}

func TestStartAlreadyRunningDoesNotCreateLocalState(t *testing.T) {
	dir := t.TempDir()
	config := DefaultConfig()
	calls := []string{}
	client := &ManagerClient{Invoke: func(_ context.Context, request []byte) ([]byte, error) {
		var decoded map[string]any
		if err := json.Unmarshal(request, &decoded); err != nil {
			t.Fatal(err)
		}
		calls = append(calls, decoded["operation"].(string))
		return responseJSON("ALREADY_RUNNING", true, "VALIDATED", map[string]any{"root": linuxTemplateRoot}), nil
	}}
	spawned := false
	app := &App{
		Manager: client,
		SpawnController: func(context.Context, string, string, string, string) (ControllerReceipt, string, error) {
			spawned = true
			return ControllerReceipt{}, "", nil
		},
		ControllerProof: controllerProof,
		Now:             func() string { return "2026-09-10T00:00:00Z" },
	}
	result := startCommand(app, dir, config)
	if result.response.Code != "ALREADY_RUNNING" || !result.response.OK {
		t.Fatalf("result = %#v", result.response)
	}
	if spawned || len(calls) != 1 || calls[0] != "probe" {
		t.Fatalf("start mutated after already-running: calls=%v spawned=%v", calls, spawned)
	}
	if _, err := os.Stat(statePath(dir)); !os.IsNotExist(err) {
		t.Fatalf("state was written: %v", err)
	}
}

func TestStartPersistsReceiptBeforeReadinessAndUsesFakeManager(t *testing.T) {
	dir := t.TempDir()
	root := linuxRunBase + "/qwenctl-20260910T010203Z-1234abcd"
	operations := []string{}
	client := &ManagerClient{Invoke: func(_ context.Context, request []byte) ([]byte, error) {
		var decoded map[string]any
		if err := json.Unmarshal(request, &decoded); err != nil {
			t.Fatal(err)
		}
		operation := decoded["operation"].(string)
		operations = append(operations, operation)
		switch operation {
		case "probe":
			return responseJSON("STOPPED", true, "VALIDATED", map[string]any{"status": "STOPPED"}), nil
		case "prepare":
			return responseJSON("PREPARED", true, "VALIDATED", map[string]any{"root": root, "launch": root + "/launch.json"}), nil
		case "await_ready":
			return responseJSON("READY", true, "VALIDATED", map[string]any{"root": root}), nil
		default:
			t.Fatalf("unexpected operation %s", operation)
			return nil, nil
		}
	}}
	app := &App{
		Manager: client,
		SpawnController: func(_ context.Context, gotRoot, launch, stdout, stderr string) (ControllerReceipt, string, error) {
			if gotRoot != root || launch != root+"/launch.json" || stdout == "" || stderr == "" {
				t.Fatalf("bad spawn inputs: %s %s %s %s", gotRoot, launch, stdout, stderr)
			}
			return ControllerReceipt{ProcessID: 9001, ReturnCode: 0}, "fixed-command", nil
		},
		ControllerProof: controllerProof,
		Now:             func() string { return "2026-09-10T00:00:00Z" },
	}
	result := startCommand(app, dir, DefaultConfig())
	if !result.response.OK || result.response.Code != "READY" {
		t.Fatalf("result = %#v", result.response)
	}
	if got, want := strings.Join(operations, ","), "probe,prepare,await_ready"; got != want {
		t.Fatalf("operations = %s, want %s", got, want)
	}
	state, exists, err := loadState(dir)
	if err != nil || !exists {
		t.Fatalf("state missing: %v", err)
	}
	if state.Phase != "active" || state.ControllerPID != 9001 || state.Root != root {
		t.Fatalf("state = %#v", state)
	}
	if !filepath.IsAbs(state.LogDir) {
		t.Fatal("log directory is not absolute")
	}
}

func TestStopRequiresControllerProofBeforeManagerStop(t *testing.T) {
	dir := t.TempDir()
	state := RunState{
		Phase:                 "active",
		Root:                  linuxRunBase + "/qwenctl-20260910T010203Z-1234abcd",
		LogDir:                filepath.Join(dir, "logs"),
		ControllerPID:         42,
		ControllerCommand:     "fixture-controller",
		ControllerFingerprint: sha256Text("fixture-controller"),
	}
	if err := writeState(dir, state); err != nil {
		t.Fatal(err)
	}
	operations := []string{}
	client := &ManagerClient{Invoke: func(_ context.Context, request []byte) ([]byte, error) {
		var decoded map[string]any
		if err := json.Unmarshal(request, &decoded); err != nil {
			t.Fatal(err)
		}
		operation := decoded["operation"].(string)
		operations = append(operations, operation)
		if operation == "probe" {
			return responseJSON("ALREADY_RUNNING", true, "VALIDATED", map[string]any{
				"root":       state.Root,
				"stop_token": map[string]any{"root": state.Root, "pid": 7, "boot_id": "b", "start_ticks": "s", "model": qwenModelName, "port": 8081},
			}), nil
		}
		return responseJSON("STOPPED", true, "VALIDATED", map[string]any{"root": state.Root}), nil
	}}
	proofCalled := false
	app := &App{
		Manager: client,
		ControllerProof: func(_ context.Context, root string, pid uint32) (map[string]any, error) {
			proofCalled = true
			if root != state.Root || pid != 42 {
				t.Fatalf("proof inputs root=%s pid=%d", root, pid)
			}
			return map[string]any{"count": float64(1), "matches": []any{map[string]any{"pid": float64(42), "command_line": "fixture-controller"}}}, nil
		},
		Now: func() string { return "2026-09-10T00:00:00Z" },
	}
	result := stopCommand(app, dir, DefaultConfig())
	if !proofCalled || result.response.Code != "STOPPED" || strings.Join(operations, ",") != "probe,stop" {
		t.Fatalf("result=%#v proof=%v operations=%v", result.response, proofCalled, operations)
	}
	updated, exists, err := loadState(dir)
	if err != nil || !exists || updated.Phase != "stopped" {
		t.Fatalf("updated state=%#v exists=%v err=%v", updated, exists, err)
	}
}

func TestLegacyTemplateReceiptPinsControllerPID(t *testing.T) {
	home := t.TempDir()
	logDir := filepath.Join(home, filepath.FromSlash(legacyLogRelative))
	if err := os.MkdirAll(logDir, 0700); err != nil {
		t.Fatal(err)
	}
	receipt := map[string]any{
		"ProcessId": uint32(36008), "ReturnValue": uint32(0),
		"Directory": logDir, "Utc": "2026-09-09T21:06:45Z",
	}
	contents, _ := json.Marshal(receipt)
	if err := os.WriteFile(filepath.Join(logDir, "receipt.json"), contents, 0600); err != nil {
		t.Fatal(err)
	}
	state, ok, err := loadLegacyTemplateState(home)
	if err != nil || !ok {
		t.Fatalf("legacy state missing: ok=%v err=%v", ok, err)
	}
	if state.ControllerPID != 36008 || state.Root != linuxTemplateRoot || state.ControllerFingerprint == "" {
		t.Fatalf("legacy state not pinned: %#v", state)
	}
}

func TestStopTokenAndControllerProofRejectUnpinnedPID(t *testing.T) {
	data := map[string]any{"stop_token": map[string]any{"root": linuxTemplateRoot}}
	if _, err := requestTokenWithControllerPID(data, 0); err == nil {
		t.Fatal("zero controller PID accepted in stop token")
	}
	if _, err := controllerProof(context.Background(), linuxTemplateRoot, 0); err == nil {
		t.Fatal("zero controller PID reached controller enumeration")
	}
}

func TestLogsTailUsesPersistedStateWithoutWrites(t *testing.T) {
	dir := t.TempDir()
	logDir := filepath.Join(dir, "logs", "run")
	if err := os.MkdirAll(logDir, 0700); err != nil {
		t.Fatal(err)
	}
	stderr := filepath.Join(logDir, "stderr.log")
	if err := os.WriteFile(stderr, []byte("one\ntwo\nthree\n"), 0600); err != nil {
		t.Fatal(err)
	}
	state := RunState{Phase: "active", Root: linuxRunBase + "/qwenctl-20260910T010203Z-1234abcd", LogDir: logDir, StdoutLog: filepath.Join(logDir, "stdout.log"), StderrLog: stderr}
	if err := writeState(dir, state); err != nil {
		t.Fatal(err)
	}
	result := logsCommand(NewApp(), dir, []string{"--lines", "2"})
	if !result.response.OK {
		t.Fatalf("logs result=%#v", result.response)
	}
	data, err := dataObject(result.response)
	if err != nil {
		t.Fatal(err)
	}
	lines := data["lines"].([]any)
	if len(lines) != 2 || lines[0] != "two" || lines[1] != "three" {
		t.Fatalf("lines=%v", lines)
	}
}
