package main

import (
	"bytes"
	"context"
	"crypto/sha256"
	_ "embed"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strings"
	"time"
	"unicode/utf16"
)

const (
	managerSchemaVersion = 1
	linuxRunBase         = "/home/operator/qwen38-5090/optimization-runs"
	linuxTemplateRoot    = linuxRunBase + "/pi-serving-059"
	qwenModelName        = "qwen3.8-27b-normal"
	legacyLogRelative    = "Documents/QwenResearch/historical/outputs/qwen38-windows/pi-serving-059"
)

//go:embed wsl_manager.py
var embeddedManagerSource string

type ManagerError struct {
	Kind    string `json:"kind"`
	Message string `json:"message"`
}

type ManagerEnvelope struct {
	SchemaVersion int             `json:"schema_version"`
	OK            bool            `json:"ok"`
	Code          string          `json:"code"`
	Profile       string          `json:"profile"`
	Data          json.RawMessage `json:"data"`
	Error         *ManagerError   `json:"error"`
}

type ManagerInvoker func(context.Context, []byte) ([]byte, error)

type ManagerClient struct {
	Invoke ManagerInvoker
}

func NewManagerClient() *ManagerClient {
	return &ManagerClient{Invoke: invokeEmbeddedManager}
}

func (m *ManagerClient) Call(ctx context.Context, operation string, config Config, fields map[string]any) (ManagerEnvelope, error) {
	if m == nil || m.Invoke == nil {
		return ManagerEnvelope{}, errors.New("cliente WSL não configurado")
	}
	if err := config.Validate(); err != nil {
		return ManagerEnvelope{}, err
	}
	request := map[string]any{
		"schema_version": managerSchemaVersion,
		"operation":      operation,
		"config":         configMap(config),
	}
	for key, value := range fields {
		request[key] = value
	}
	encoded, err := json.Marshal(request)
	if err != nil {
		return ManagerEnvelope{}, fmt.Errorf("request WSL inválido: %w", err)
	}
	output, err := m.Invoke(ctx, encoded)
	if err != nil {
		return ManagerEnvelope{}, err
	}
	var response ManagerEnvelope
	if err := json.Unmarshal(output, &response); err != nil {
		return ManagerEnvelope{}, fmt.Errorf("resposta WSL não é JSON: %w", err)
	}
	if response.SchemaVersion != managerSchemaVersion || response.Code == "" {
		return ManagerEnvelope{}, errors.New("resposta WSL fora do protocolo")
	}
	if len(response.Data) == 0 || bytes.Equal(response.Data, []byte("null")) {
		response.Data = json.RawMessage(`{}`)
	}
	return response, nil
}

func embeddedPythonProgram(request []byte) []byte {
	// python3 - receives one self-contained wrapper.  The manager source is
	// executed under a non-main module name, so its command-line footer does not
	// run; the request is then passed to its public protocol function.  This
	// avoids temporary files and keeps read-only operations write-free.
	sourceJSON, _ := json.Marshal(embeddedManagerSource)
	requestJSON, _ := json.Marshal(string(request))
	var builder strings.Builder
	builder.WriteString("import io\n")
	builder.WriteString("_qwenctl_ns={'__name__':'qwenctl_embedded'}\n")
	builder.WriteString("exec(compile(")
	builder.Write(sourceJSON)
	builder.WriteString(", 'wsl_manager.py', 'exec'), _qwenctl_ns)\n")
	builder.WriteString("_qwenctl_ns['run_protocol'](io.StringIO(")
	builder.Write(requestJSON)
	builder.WriteString("))\n")
	return []byte(builder.String())
}

func invokeEmbeddedManager(ctx context.Context, request []byte) ([]byte, error) {
	command := exec.CommandContext(ctx, "wsl.exe", "-d", "Ubuntu", "--", "python3", "-")
	command.Stdin = bytes.NewReader(embeddedPythonProgram(request))
	var stdout bytes.Buffer
	var stderr bytes.Buffer
	command.Stdout = &stdout
	command.Stderr = &stderr
	if err := command.Run(); err != nil {
		diagnostic := strings.TrimSpace(stderr.String())
		if diagnostic != "" {
			return nil, fmt.Errorf("WSL manager falhou: %w: %s", err, diagnostic)
		}
		return nil, fmt.Errorf("WSL manager falhou: %w", err)
	}
	return bytes.TrimSpace(stdout.Bytes()), nil
}

type RunState struct {
	SchemaVersion         int    `json:"schema_version"`
	Phase                 string `json:"phase"`
	Root                  string `json:"root"`
	LaunchPath            string `json:"launch_path"`
	LogDir                string `json:"log_dir"`
	StdoutLog             string `json:"stdout_log"`
	StderrLog             string `json:"stderr_log"`
	ControllerPID         uint32 `json:"controller_pid"`
	ControllerReturn      uint32 `json:"controller_return_value"`
	ControllerCommand     string `json:"controller_command"`
	ControllerFingerprint string `json:"controller_sha256"`
	StartedAt             string `json:"started_at"`
	ReadyAt               string `json:"ready_at,omitempty"`
	StoppedAt             string `json:"stopped_at,omitempty"`
}

type legacyControllerReceipt struct {
	ProcessID   uint32 `json:"ProcessId"`
	ReturnValue uint32 `json:"ReturnValue"`
	Directory   string `json:"Directory"`
	UTC         string `json:"Utc"`
}

func loadLegacyTemplateState(userHome string) (RunState, bool, error) {
	if userHome == "" || !filepath.IsAbs(userHome) {
		return RunState{}, false, errors.New("perfil Windows inválido para recibo legado")
	}
	logDir := filepath.Join(userHome, filepath.FromSlash(legacyLogRelative))
	receiptPath := filepath.Join(logDir, "receipt.json")
	info, err := os.Lstat(receiptPath)
	if errors.Is(err, os.ErrNotExist) {
		return RunState{}, false, errors.New("recibo do controller validado não foi encontrado")
	}
	if err != nil {
		return RunState{}, false, fmt.Errorf("não foi possível ler recibo do controller validado: %w", err)
	}
	if info.Mode()&os.ModeSymlink != 0 || !info.Mode().IsRegular() {
		return RunState{}, false, errors.New("recibo do controller validado não é arquivo regular")
	}
	contents, err := os.ReadFile(receiptPath)
	if err != nil {
		return RunState{}, false, fmt.Errorf("não foi possível ler recibo do controller validado: %w", err)
	}
	var receipt legacyControllerReceipt
	if err := json.Unmarshal(contents, &receipt); err != nil {
		return RunState{}, false, fmt.Errorf("recibo do controller validado é inválido: %w", err)
	}
	if receipt.ProcessID == 0 || receipt.ReturnValue != 0 || !strings.EqualFold(filepath.Clean(receipt.Directory), filepath.Clean(logDir)) {
		return RunState{}, false, errors.New("recibo do controller validado não corresponde ao diretório esperado")
	}
	launchPath := linuxTemplateRoot + "/launch.json"
	stdoutLog := filepath.Join(logDir, "stdout.log")
	stderrLog := filepath.Join(logDir, "stderr.log")
	command, err := buildControllerCommand(linuxTemplateRoot, launchPath, stdoutLog, stderrLog)
	if err != nil {
		return RunState{}, false, err
	}
	return RunState{
		Phase:                 "active",
		Root:                  linuxTemplateRoot,
		LaunchPath:            launchPath,
		LogDir:                logDir,
		StdoutLog:             stdoutLog,
		StderrLog:             stderrLog,
		ControllerPID:         receipt.ProcessID,
		ControllerReturn:      receipt.ReturnValue,
		ControllerCommand:     command,
		ControllerFingerprint: sha256Text(command),
		StartedAt:             receipt.UTC,
	}, true, nil
}

func loadState(dir string) (RunState, bool, error) {
	path := statePath(dir)
	if info, statErr := os.Lstat(path); statErr == nil && info.Mode()&os.ModeSymlink != 0 {
		return RunState{}, false, errors.New("state.json symlink não é aceito")
	}
	contents, err := os.ReadFile(path)
	if errors.Is(err, os.ErrNotExist) {
		return RunState{}, false, nil
	}
	if err != nil {
		return RunState{}, false, fmt.Errorf("não foi possível ler state.json: %w", err)
	}
	var state RunState
	if err := json.Unmarshal(contents, &state); err != nil {
		return RunState{}, false, fmt.Errorf("state.json inválido: %w", err)
	}
	if state.SchemaVersion != managerSchemaVersion || state.Root == "" || state.LogDir == "" {
		return RunState{}, false, errors.New("state.json não contém recibo completo")
	}
	return state, true, nil
}

func writeState(dir string, state RunState) error {
	state.SchemaVersion = managerSchemaVersion
	return writeAtomicJSON(statePath(dir), state)
}

func managedLinuxRoot(root string) bool {
	if root == linuxTemplateRoot {
		return true
	}
	prefix := linuxRunBase + "/qwenctl-"
	if !strings.HasPrefix(root, prefix) || strings.Contains(root[len(prefix):], "/") {
		return false
	}
	name := strings.TrimPrefix(root, prefix)
	matched, _ := regexp.MatchString(`^[0-9]{8}T[0-9]{6}Z-[A-Za-z0-9]{8}$`, name)
	return matched
}

func safeWindowsText(value string) bool {
	return !strings.ContainsAny(value, "\r\n\"")
}

func buildControllerCommand(root, launchPath, stdoutLog, stderrLog string) (string, error) {
	if !managedLinuxRoot(root) {
		return "", errors.New("root Linux não é gerenciado")
	}
	expectedLaunch := strings.TrimRight(root, "/") + "/launch.json"
	if launchPath != expectedLaunch {
		return "", errors.New("launch.json não pertence ao root")
	}
	for _, value := range []string{stdoutLog, stderrLog} {
		if value == "" || !safeWindowsText(value) {
			return "", errors.New("caminho de log inválido")
		}
	}
	return fmt.Sprintf(
		`C:\Windows\System32\cmd.exe /d /c C:\Windows\System32\wsl.exe -d Ubuntu -- python3 %s/exec-manifest.py %s 1>"%s" 2>"%s"`,
		strings.TrimRight(root, "/"), launchPath, stdoutLog, stderrLog,
	), nil
}

func powershellQuote(value string) string {
	return "'" + strings.ReplaceAll(value, "'", "''") + "'"
}

func utf16LEBase64(value string) string {
	encoded := utf16.Encode([]rune(value))
	bytesValue := make([]byte, 0, len(encoded)*2)
	for _, unit := range encoded {
		bytesValue = append(bytesValue, byte(unit), byte(unit>>8))
	}
	return base64.StdEncoding.EncodeToString(bytesValue)
}

type ControllerReceipt struct {
	ProcessID  uint32 `json:"ProcessId"`
	ReturnCode uint32 `json:"ReturnValue"`
}

func powershellExecutable() string {
	return "powershell.exe"
}

func runPowerShellJSON(ctx context.Context, script string) ([]byte, error) {
	encoded := utf16LEBase64(script)
	command := exec.CommandContext(
		ctx,
		powershellExecutable(),
		"-NoLogo",
		"-NoProfile",
		"-NonInteractive",
		"-ExecutionPolicy",
		"Bypass",
		"-EncodedCommand",
		encoded,
	)
	var stdout bytes.Buffer
	var stderr bytes.Buffer
	command.Stdout = &stdout
	command.Stderr = &stderr
	if err := command.Run(); err != nil {
		message := strings.TrimSpace(stderr.String())
		if message != "" {
			return nil, fmt.Errorf("PowerShell falhou: %w: %s", err, message)
		}
		return nil, fmt.Errorf("PowerShell falhou: %w", err)
	}
	return lastJSONLine(stdout.Bytes())
}

func lastJSONLine(output []byte) ([]byte, error) {
	lines := strings.Split(strings.ReplaceAll(string(output), "\r\n", "\n"), "\n")
	for index := len(lines) - 1; index >= 0; index-- {
		line := strings.TrimSpace(lines[index])
		if strings.HasPrefix(line, "{") || strings.HasPrefix(line, "[") {
			return []byte(line), nil
		}
	}
	return nil, errors.New("PowerShell não retornou JSON")
}

func spawnController(ctx context.Context, root, launchPath, stdoutLog, stderrLog string) (ControllerReceipt, string, error) {
	controllerCommand, err := buildControllerCommand(root, launchPath, stdoutLog, stderrLog)
	if err != nil {
		return ControllerReceipt{}, "", err
	}
	currentDirectory := filepath.Dir(stdoutLog)
	script := "$ErrorActionPreference='Stop';$ProgressPreference='SilentlyContinue';" +
		"$command=" + powershellQuote(controllerCommand) + ";" +
		"$current=" + powershellQuote(currentDirectory) + ";" +
		"$startup=New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{ShowWindow=[uint16]0};" +
		"$result=Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine=$command;CurrentDirectory=$current;ProcessStartupInformation=$startup};" +
		"[Console]::Out.WriteLine((@{ProcessId=[uint32]$result.ProcessId;ReturnValue=[uint32]$result.ReturnValue}|ConvertTo-Json -Compress))"
	output, err := runPowerShellJSON(ctx, script)
	if err != nil {
		return ControllerReceipt{}, controllerCommand, err
	}
	var receipt ControllerReceipt
	if err := json.Unmarshal(output, &receipt); err != nil {
		return ControllerReceipt{}, controllerCommand, fmt.Errorf("recibo CIM inválido: %w", err)
	}
	return receipt, controllerCommand, nil
}

func controllerProof(ctx context.Context, root string, expectedPID uint32) (map[string]any, error) {
	if !managedLinuxRoot(root) {
		return nil, errors.New("root Linux não é gerenciado")
	}
	if expectedPID == 0 {
		return nil, errors.New("PID do controller não está pinado")
	}
	script := "$ErrorActionPreference='Stop';$ProgressPreference='SilentlyContinue';" +
		"$needle=" + powershellQuote(root+"/exec-manifest.py") + ";$launch=" + powershellQuote(root+"/launch.json") + ";" +
		"$matches=@(Get-CimInstance -ClassName Win32_Process | Where-Object { $_.Name -ieq 'cmd.exe' -and $_.CommandLine -and $_.CommandLine.Contains($needle) -and $_.CommandLine.Contains($launch) });" +
		"$items=@();foreach($item in $matches){$items += [ordered]@{pid=[uint32]$item.ProcessId;name=[string]$item.Name;command_line=[string]$item.CommandLine}};" +
		"[ordered]@{count=$items.Count;matches=$items}|ConvertTo-Json -Compress -Depth 4"
	output, err := runPowerShellJSON(ctx, script)
	if err != nil {
		return nil, err
	}
	var proof map[string]any
	if err := json.Unmarshal(output, &proof); err != nil {
		return nil, fmt.Errorf("prova de controller inválida: %w", err)
	}
	if count, ok := proof["count"].(float64); !ok || int(count) != 1 {
		return proof, errors.New("não foi provado exatamente um controller")
	}
	if expectedPID != 0 {
		matches := controllerMatches(proof["matches"])
		if len(matches) != 1 {
			return proof, errors.New("prova de controller sem match único")
		}
		match := matches[0]
		pid, ok := match["pid"].(float64)
		if !ok || uint32(pid) != expectedPID {
			return proof, errors.New("controller não corresponde ao recibo persistido")
		}
	}
	return proof, nil
}

func controllerMatches(value any) []map[string]any {
	switch typed := value.(type) {
	case []any:
		matches := make([]map[string]any, 0, len(typed))
		for _, item := range typed {
			if match, ok := item.(map[string]any); ok {
				matches = append(matches, match)
			}
		}
		return matches
	case map[string]any:
		return []map[string]any{typed}
	default:
		return nil
	}
}

func dataObject(response ManagerEnvelope) (map[string]any, error) {
	result := map[string]any{}
	if len(response.Data) == 0 || bytes.Equal(response.Data, []byte("null")) {
		return result, nil
	}
	if err := json.Unmarshal(response.Data, &result); err != nil {
		return nil, fmt.Errorf("data WSL inválido: %w", err)
	}
	return result, nil
}

func stringData(data map[string]any, key string) (string, error) {
	value, ok := data[key].(string)
	if !ok || value == "" {
		return "", fmt.Errorf("resposta WSL sem %s", key)
	}
	return value, nil
}

func nowRFC3339() string {
	return time.Now().UTC().Format(time.RFC3339Nano)
}

func sha256Text(value string) string {
	sum := sha256.Sum256([]byte(value))
	return fmt.Sprintf("%x", sum[:])
}

func rawMap(value map[string]any) json.RawMessage {
	encoded, _ := json.Marshal(value)
	return encoded
}

func controllerPIDFromState(state RunState, root string) uint32 {
	if state.Root != root {
		return 0
	}
	return state.ControllerPID
}

func requestTokenWithControllerPID(data map[string]any, controllerPID uint32) (map[string]any, error) {
	raw, ok := data["stop_token"].(map[string]any)
	if !ok {
		return nil, errors.New("resposta WSL sem stop token")
	}
	if controllerPID == 0 {
		return nil, errors.New("stop recusado sem PID pinado do controller")
	}
	copyToken := make(map[string]any, len(raw)+1)
	for key, value := range raw {
		copyToken[key] = value
	}
	copyToken["controller_pid"] = controllerPID
	return copyToken, nil
}

func managerFailure(response ManagerEnvelope) error {
	if response.Error != nil {
		return fmt.Errorf("%s: %s", response.Error.Kind, response.Error.Message)
	}
	return fmt.Errorf("operação WSL recusada: %s", response.Code)
}
