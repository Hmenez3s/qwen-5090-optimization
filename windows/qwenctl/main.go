package main

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"os/signal"
	"path/filepath"
	"strconv"
	"strings"
	"syscall"
	"time"
)

var version = "0.1.0"

type commandResult struct {
	response  ManagerEnvelope
	exitCode  int
	human     string
	raw       bool
	follow    *followSpec
	logStream string
}

type App struct {
	Manager         *ManagerClient
	SpawnController func(context.Context, string, string, string, string) (ControllerReceipt, string, error)
	ControllerProof func(context.Context, string, uint32) (map[string]any, error)
	Now             func() string
}

func NewApp() *App {
	return &App{
		Manager:         NewManagerClient(),
		SpawnController: spawnController,
		ControllerProof: controllerProof,
		Now:             nowRFC3339,
	}
}

func successEnvelope(code, profile string, data any) ManagerEnvelope {
	encoded, _ := json.Marshal(data)
	return ManagerEnvelope{
		SchemaVersion: managerSchemaVersion,
		OK:            true,
		Code:          code,
		Profile:       profile,
		Data:          encoded,
	}
}

func failureEnvelope(code, profile, kind, message string, data any) ManagerEnvelope {
	encoded, _ := json.Marshal(data)
	return ManagerEnvelope{
		SchemaVersion: managerSchemaVersion,
		OK:            false,
		Code:          code,
		Profile:       profile,
		Data:          encoded,
		Error:         &ManagerError{Kind: kind, Message: message},
	}
}

func responseExitCode(response ManagerEnvelope) int {
	if response.OK {
		return 0
	}
	switch response.Code {
	case "INVALID_CONFIG", "INVALID_REQUEST", "SYNTAX":
		return 2
	case "BLOCKED", "BUSY":
		return 3
	case "NOT_READY", "NOT_RUNNING", "UNHEALTHY":
		return 4
	default:
		return 1
	}
}

func localFailure(profile, code, kind, message string) commandResult {
	response := failureEnvelope(code, profile, kind, message, map[string]any{})
	return commandResult{response: response, exitCode: responseExitCode(response)}
}

func managerResult(response ManagerEnvelope) commandResult {
	return commandResult{response: response, exitCode: responseExitCode(response)}
}

func withData(response ManagerEnvelope, additions map[string]any) ManagerEnvelope {
	data, err := dataObject(response)
	if err != nil {
		return response
	}
	for key, value := range additions {
		data[key] = value
	}
	response.Data = rawMap(data)
	return response
}

func appNow(app *App) string {
	if app != nil && app.Now != nil {
		return app.Now()
	}
	return nowRFC3339()
}

func callManager(ctx context.Context, app *App, operation string, config Config, fields map[string]any) (ManagerEnvelope, error) {
	if app == nil || app.Manager == nil {
		return ManagerEnvelope{}, errors.New("cliente WSL não configurado")
	}
	return app.Manager.Call(ctx, operation, config, fields)
}

func contextFor(seconds int64) (context.Context, context.CancelFunc) {
	if seconds < 30 {
		seconds = 30
	}
	return context.WithTimeout(context.Background(), time.Duration(seconds)*time.Second)
}

func startCommand(app *App, dir string, config Config) commandResult {
	profile := config.Profile()
	probeContext, cancel := contextFor(45)
	probe, err := callManager(probeContext, app, "probe", config, nil)
	cancel()
	if err != nil {
		return localFailure(profile, "ERROR", "wsl", err.Error())
	}
	if probe.Code == "ALREADY_RUNNING" {
		// This is deliberately before every local mkdir/state write.  A healthy
		// template or managed run is preserved byte-for-byte.
		return managerResult(probe)
	}
	if !probe.OK && probe.Code != "STOPPED" {
		return managerResult(probe)
	}

	prepareContext, cancel := contextFor(90)
	prepared, err := callManager(prepareContext, app, "prepare", config, nil)
	cancel()
	if err != nil {
		return localFailure(profile, "ERROR", "wsl", err.Error())
	}
	if prepared.Code == "ALREADY_RUNNING" {
		return managerResult(prepared)
	}
	if !prepared.OK {
		return managerResult(prepared)
	}
	data, err := dataObject(prepared)
	if err != nil {
		return localFailure(profile, "ERROR", "protocol", err.Error())
	}
	root, err := stringData(data, "root")
	if err != nil || !managedLinuxRoot(root) {
		return localFailure(profile, "ERROR", "protocol", "root preparado inválido")
	}
	launchPath := root + "/launch.json"
	if value, ok := data["launch"].(string); ok {
		if value != launchPath {
			return localFailure(profile, "ERROR", "protocol", "launch.json preparado inválido")
		}
		launchPath = value
	}

	logParent := filepath.Join(dir, "logs")
	if err := os.MkdirAll(logParent, 0700); err != nil {
		return localFailure(profile, "ERROR", "storage", fmt.Sprintf("não foi possível criar logs: %v", err))
	}
	logDir := filepath.Join(logParent, filepath.Base(root))
	if err := os.Mkdir(logDir, 0700); err != nil {
		return localFailure(profile, "ERROR", "storage", fmt.Sprintf("diretório de logs não é exclusivo: %v", err))
	}
	stdoutLog := filepath.Join(logDir, "stdout.log")
	stderrLog := filepath.Join(logDir, "stderr.log")
	controllerCommand, err := buildControllerCommand(root, launchPath, stdoutLog, stderrLog)
	if err != nil {
		return localFailure(profile, "ERROR", "controller", err.Error())
	}
	state := RunState{
		Phase:                 "prepared",
		Root:                  root,
		LaunchPath:            launchPath,
		LogDir:                logDir,
		StdoutLog:             stdoutLog,
		StderrLog:             stderrLog,
		ControllerCommand:     controllerCommand,
		ControllerFingerprint: sha256Text(controllerCommand),
		StartedAt:             appNow(app),
	}
	if err := writeState(dir, state); err != nil {
		return localFailure(profile, "ERROR", "storage", fmt.Sprintf("não foi possível persistir ownership: %v", err))
	}

	spawn := app.SpawnController
	if spawn == nil {
		return localFailure(profile, "ERROR", "controller", "controller fake/real não está configurado")
	}
	spawnContext, spawnCancel := contextFor(30)
	receipt, actualCommand, err := spawn(spawnContext, root, launchPath, stdoutLog, stderrLog)
	spawnCancel()
	if actualCommand != "" {
		state.ControllerCommand = actualCommand
		state.ControllerFingerprint = sha256Text(actualCommand)
	}
	if err != nil {
		state.Phase = "controller_failed"
		_ = writeState(dir, state)
		return localFailure(profile, "ERROR", "controller", err.Error())
	}
	state.ControllerPID = receipt.ProcessID
	state.ControllerReturn = receipt.ReturnCode
	state.Phase = "starting"
	if receipt.ReturnCode != 0 || receipt.ProcessID == 0 {
		state.Phase = "controller_failed"
		_ = writeState(dir, state)
		return localFailure(profile, "ERROR", "controller", "Win32_Process.Create recusou o controller")
	}
	if err := writeState(dir, state); err != nil {
		return localFailure(profile, "ERROR", "storage", fmt.Sprintf("controller criado mas ownership não pôde ser atualizado: %v", err))
	}

	waitContext, cancel := contextFor(config.StartupTimeout + 45)
	ready, err := callManager(waitContext, app, "await_ready", config, map[string]any{"root": root})
	cancel()
	if err != nil {
		return localFailure(profile, "ERROR", "wsl", err.Error())
	}
	if !ready.OK {
		return managerResult(withData(ready, map[string]any{
			"root":       root,
			"log_dir":    logDir,
			"stdout_log": stdoutLog,
			"stderr_log": stderrLog,
		}))
	}
	state.Phase = "active"
	state.ReadyAt = appNow(app)
	if err := writeState(dir, state); err != nil {
		return localFailure(profile, "ERROR", "storage", fmt.Sprintf("serviço pronto mas state.json não pôde ser atualizado: %v", err))
	}
	return managerResult(withData(ready, map[string]any{
		"root":            root,
		"log_dir":         logDir,
		"stdout_log":      stdoutLog,
		"stderr_log":      stderrLog,
		"profile":         profile,
		"pending_restart": false,
	}))
}

func stopCommand(app *App, dir string, config Config) commandResult {
	profile := config.Profile()
	probeContext, cancel := contextFor(45)
	probe, err := callManager(probeContext, app, "probe", config, nil)
	cancel()
	if err != nil {
		return localFailure(profile, "ERROR", "wsl", err.Error())
	}
	if probe.Code == "STOPPED" {
		return managerResult(probe)
	}
	if probe.Code != "ALREADY_RUNNING" || !probe.OK {
		return managerResult(probe)
	}
	data, err := dataObject(probe)
	if err != nil {
		return localFailure(profile, "ERROR", "protocol", err.Error())
	}
	root, err := stringData(data, "root")
	if err != nil || !managedLinuxRoot(root) {
		return localFailure(profile, "BLOCKED", "identity", "root ativo não é gerenciado")
	}
	state, hasState, stateErr := loadState(dir)
	if stateErr != nil {
		return localFailure(profile, "BLOCKED", "storage", stateErr.Error())
	}
	if !hasState && root == linuxTemplateRoot {
		userHome, homeErr := os.UserHomeDir()
		if homeErr != nil {
			return localFailure(profile, "BLOCKED", "controller", "perfil Windows indisponível para validar controller")
		}
		state, hasState, stateErr = loadLegacyTemplateState(userHome)
		if stateErr != nil {
			return localFailure(profile, "BLOCKED", "controller", stateErr.Error())
		}
	}
	if !hasState {
		return localFailure(profile, "BLOCKED", "controller", "stop recusado sem recibo persistido do controller")
	}
	if (state.Phase == "active" || state.Phase == "starting") && state.Root != root {
		return localFailure(profile, "BLOCKED", "identity", "state.json aponta para outro root ativo")
	}
	controllerPID := controllerPIDFromState(state, root)
	if controllerPID == 0 {
		return localFailure(profile, "BLOCKED", "controller", "stop recusado sem PID pinado do controller")
	}
	if app.ControllerProof == nil {
		return localFailure(profile, "BLOCKED", "controller", "prova de controller não está configurada")
	}
	proofContext, cancel := contextFor(45)
	proof, err := app.ControllerProof(proofContext, root, controllerPID)
	cancel()
	if err != nil {
		return localFailure(profile, "BLOCKED", "controller", err.Error())
	}
	if hasState && (state.Phase == "active" || state.Phase == "starting") {
		matches := controllerMatches(proof["matches"])
		if state.ControllerCommand == "" || len(matches) != 1 {
			return localFailure(profile, "BLOCKED", "controller", "state.json não contém prova de controller completa")
		}
		command, ok := matches[0]["command_line"].(string)
		if !ok || strings.TrimSpace(command) != strings.TrimSpace(state.ControllerCommand) || sha256Text(command) != state.ControllerFingerprint {
			return localFailure(profile, "BLOCKED", "controller", "controller atual não corresponde ao recibo persistido")
		}
	}
	token, err := requestTokenWithControllerPID(data, controllerPID)
	if err != nil {
		return localFailure(profile, "BLOCKED", "identity", err.Error())
	}
	stopContext, cancel := contextFor(180)
	stopped, err := callManager(stopContext, app, "stop", config, map[string]any{
		"stop_token": token,
		"controller": proof,
	})
	cancel()
	if err != nil {
		return localFailure(profile, "ERROR", "wsl", err.Error())
	}
	if stopped.OK && stopped.Code == "STOPPED" && hasState {
		state.Phase = "stopped"
		state.StoppedAt = appNow(app)
		if err := writeState(dir, state); err != nil {
			// The signal already happened.  Do not retry or clean up; report the
			// durable ownership gap as an error for the orchestrator/user.
			return localFailure(profile, "ERROR", "storage", fmt.Sprintf("serviço parado mas state.json não pôde ser atualizado: %v", err))
		}
	}
	return managerResult(stopped)
}

func statusCommand(app *App, config Config) commandResult {
	ctx, cancel := contextFor(45)
	response, err := callManager(ctx, app, "probe", config, nil)
	cancel()
	if err != nil {
		return localFailure(config.Profile(), "ERROR", "wsl", err.Error())
	}
	return managerResult(response)
}

func healthCommand(app *App, dir string, config Config) commandResult {
	fields := map[string]any{}
	state, hasState, err := loadState(dir)
	if err != nil {
		return localFailure(config.Profile(), "BLOCKED", "storage", err.Error())
	}
	if hasState && (state.Phase == "active" || state.Phase == "starting") && managedLinuxRoot(state.Root) {
		fields["root"] = state.Root
	}
	ctx, cancel := contextFor(90)
	response, err := callManager(ctx, app, "health", config, fields)
	cancel()
	if err != nil {
		return localFailure(config.Profile(), "ERROR", "wsl", err.Error())
	}
	return managerResult(response)
}

type parsedArgs struct {
	command string
	args    []string
	jsonOut bool
}

func parseArgs(args []string) (parsedArgs, error) {
	if len(args) == 0 {
		return parsedArgs{command: "help"}, nil
	}
	result := parsedArgs{command: args[0]}
	for _, arg := range args[1:] {
		if arg == "--json" {
			if result.jsonOut {
				return parsedArgs{}, errors.New("--json repetido")
			}
			result.jsonOut = true
			continue
		}
		result.args = append(result.args, arg)
	}
	return result, nil
}

func helpText() string {
	return `qwenctl — controle seguro do serviço Qwen3.8 NORMAL no gpu-host WSL

Uso:
  qwenctl start [--json]       inicia somente um root novo, se necessário
  qwenctl stop [--json]        para após provar identidade, controller e ociosidade
  qwenctl status [--json]      mostra processo/API sem escrever
  qwenctl health [--json]      verifica identidade, perfil, hooks, AOT, KDA e potência
  qwenctl config show|get|set|reset
  qwenctl logs [--lines N] [--follow] [--stdout|--stderr]
  qwenctl version

Configuração permitida (aplica no próximo start):
  context-length 131072..131200 (padrão 131200)
  max-total-tokens >= context-length e <=131328 (padrão 131328)
  random-seed 0..2147483647 (padrão 7319)
  startup-timeout-seconds 60..1800 (padrão 900)

Não há modo force, argumentos livres, download, túnel, serviço ou tarefa agendada.`
}

func versionCommand(jsonOut bool) commandResult {
	response := successEnvelope("VERSION", "VALIDATED", map[string]any{"version": version})
	return commandResult{response: response, exitCode: 0, human: "qwenctl " + version}
}

func configCommand(dir string, args []string) commandResult {
	if len(args) == 0 {
		return localFailure("VALIDATED", "SYNTAX", "syntax", "use config show|get|set|reset")
	}
	subcommand := args[0]
	switch subcommand {
	case "show":
		if len(args) != 1 {
			return localFailure("VALIDATED", "SYNTAX", "syntax", "config show não aceita argumentos")
		}
		config, err := loadConfig(dir)
		if err != nil {
			return localFailure("VALIDATED", "INVALID_CONFIG", "config", err.Error())
		}
		return managerResult(successEnvelope("CONFIG", config.Profile(), configView(config)))
	case "get":
		if len(args) != 2 || !parseConfigKey(args[1]) {
			return localFailure("VALIDATED", "SYNTAX", "syntax", "use config get <chave permitida>")
		}
		config, err := loadConfig(dir)
		if err != nil {
			return localFailure("VALIDATED", "INVALID_CONFIG", "config", err.Error())
		}
		view := map[string]any{"key": args[1], "value": configMap(config)[args[1]], "profile": config.Profile()}
		return managerResult(successEnvelope("CONFIG", config.Profile(), view))
	case "set":
		if len(args) != 3 && len(args) != 2 {
			return localFailure("VALIDATED", "SYNTAX", "syntax", "use config set <chave> <valor> ou chave=valor")
		}
		key, value := "", ""
		if len(args) == 3 {
			key, value = args[1], args[2]
		} else {
			parts := strings.SplitN(args[1], "=", 2)
			if len(parts) != 2 {
				return localFailure("VALIDATED", "SYNTAX", "syntax", "use chave=valor")
			}
			key, value = parts[0], parts[1]
		}
		if !parseConfigKey(key) {
			return localFailure("VALIDATED", "INVALID_CONFIG", "config", "chave não permitida")
		}
		config, err := loadConfig(dir)
		if err != nil {
			return localFailure("VALIDATED", "INVALID_CONFIG", "config", err.Error())
		}
		updated, err := setConfigValue(config, key, value)
		if err != nil {
			return localFailure(config.Profile(), "INVALID_CONFIG", "config", err.Error())
		}
		if err := writeConfig(dir, updated); err != nil {
			return localFailure(updated.Profile(), "ERROR", "storage", fmt.Sprintf("não foi possível gravar config.json: %v", err))
		}
		view := configView(updated)
		view["applies_next_start"] = true
		return managerResult(successEnvelope("CONFIG_UPDATED", updated.Profile(), view))
	case "reset":
		if len(args) != 1 {
			return localFailure("VALIDATED", "SYNTAX", "syntax", "config reset não aceita argumentos")
		}
		config := DefaultConfig()
		if err := writeConfig(dir, config); err != nil {
			return localFailure(config.Profile(), "ERROR", "storage", fmt.Sprintf("não foi possível gravar config.json: %v", err))
		}
		return managerResult(successEnvelope("CONFIG_RESET", config.Profile(), configView(config)))
	default:
		return localFailure("VALIDATED", "SYNTAX", "syntax", "subcomando config desconhecido")
	}
}

func pathInside(parent, child string) bool {
	rel, err := filepath.Rel(filepath.Clean(parent), filepath.Clean(child))
	return err == nil && rel != ".." && !strings.HasPrefix(rel, ".."+string(filepath.Separator)) && !filepath.IsAbs(rel)
}

const historicalLogDirectory = `C:\Users\operator\Documents\QwenResearch\historical\outputs\qwen38-windows\pi-serving-059`

func logPath(dir, stream string) (string, error) {
	state, exists, err := loadState(dir)
	if err != nil {
		return "", err
	}
	name := "stderr.log"
	if stream == "stdout" {
		name = "stdout.log"
	}
	if exists {
		if state.Root != linuxTemplateRoot && !managedLinuxRoot(state.Root) {
			return "", errors.New("state.json contém root não gerenciado")
		}
		expected := filepath.Join(state.LogDir, name)
		actual := state.StderrLog
		if stream == "stdout" {
			actual = state.StdoutLog
		}
		if filepath.Clean(actual) != filepath.Clean(expected) || !pathInside(dir, state.LogDir) {
			return "", errors.New("state.json contém caminho de log fora da área local")
		}
		if info, statErr := os.Lstat(state.LogDir); statErr == nil && info.Mode()&os.ModeSymlink != 0 {
			return "", errors.New("diretório de logs symlink não é aceito")
		}
		return actual, nil
	}
	return filepath.Join(historicalLogDirectory, name), nil
}

type logOptions struct {
	lines  int
	follow bool
	stream string
}

func parseLogOptions(args []string) (logOptions, error) {
	options := logOptions{lines: 100, stream: "stderr"}
	for index := 0; index < len(args); index++ {
		switch args[index] {
		case "--lines":
			if index+1 >= len(args) {
				return logOptions{}, errors.New("--lines requer um número")
			}
			value, err := strconv.Atoi(args[index+1])
			if err != nil || value <= 0 || value > 1000000 {
				return logOptions{}, errors.New("--lines deve estar entre 1 e 1000000")
			}
			options.lines = value
			index++
		case "--follow":
			options.follow = true
		case "--stdout":
			if options.stream == "stderr" && containsArg(args, "--stderr") {
				return logOptions{}, errors.New("escolha stdout ou stderr, não ambos")
			}
			options.stream = "stdout"
		case "--stderr":
			if options.stream == "stdout" {
				return logOptions{}, errors.New("escolha stdout ou stderr, não ambos")
			}
			options.stream = "stderr"
		default:
			return logOptions{}, fmt.Errorf("opção de logs desconhecida: %s", args[index])
		}
	}
	return options, nil
}

func containsArg(args []string, wanted string) bool {
	for _, arg := range args {
		if arg == wanted {
			return true
		}
	}
	return false
}

func readLastLines(path string, count int) ([]string, error) {
	if info, err := os.Lstat(path); err != nil {
		return nil, err
	} else if info.Mode()&os.ModeSymlink != 0 {
		return nil, errors.New("arquivo de log symlink não é aceito")
	}
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer file.Close()
	scanner := bufio.NewScanner(file)
	scanner.Buffer(make([]byte, 4096), 16*1024*1024)
	lines := make([]string, 0, count)
	for scanner.Scan() {
		if len(lines) == count {
			copy(lines, lines[1:])
			lines = lines[:count-1]
		}
		lines = append(lines, scanner.Text())
	}
	if err := scanner.Err(); err != nil {
		return nil, err
	}
	return lines, nil
}

type followSpec struct {
	path   string
	stream string
	offset int64
}

func logsCommand(app *App, dir string, args []string) commandResult {
	options, err := parseLogOptions(args)
	if err != nil {
		return localFailure("VALIDATED", "SYNTAX", "syntax", err.Error())
	}
	path, err := logPath(dir, options.stream)
	if err != nil {
		return localFailure("VALIDATED", "BLOCKED", "storage", err.Error())
	}
	if _, exists, stateErr := loadState(dir); stateErr != nil {
		return localFailure("VALIDATED", "BLOCKED", "storage", stateErr.Error())
	} else if !exists {
		if app == nil || app.ControllerProof == nil {
			return localFailure("VALIDATED", "BLOCKED", "controller", "logs históricos exigem prova do controller")
		}
		ctx, cancel := contextFor(45)
		proof, proofErr := app.ControllerProof(ctx, linuxTemplateRoot, 0)
		cancel()
		if proofErr != nil {
			return localFailure("VALIDATED", "BLOCKED", "controller", proofErr.Error())
		}
		matches := controllerMatches(proof["matches"])
		if len(matches) != 1 {
			return localFailure("VALIDATED", "BLOCKED", "controller", "controller histórico não é único")
		}
		command, ok := matches[0]["command_line"].(string)
		if !ok || !strings.Contains(command, `1>"`+filepath.Join(historicalLogDirectory, "stdout.log")+`"`) || !strings.Contains(command, `2>"`+filepath.Join(historicalLogDirectory, "stderr.log")+`"`) {
			return localFailure("VALIDATED", "BLOCKED", "controller", "redirecionamento histórico não foi provado")
		}
	}
	lines, err := readLastLines(path, options.lines)
	if err != nil {
		return localFailure("VALIDATED", "ERROR", "logs", fmt.Sprintf("não foi possível ler %s: %v", path, err))
	}
	data := map[string]any{"stream": options.stream, "path": path, "lines": lines, "following": options.follow}
	result := commandResult{
		response:  successEnvelope("LOGS", "VALIDATED", data),
		exitCode:  0,
		raw:       true,
		logStream: options.stream,
	}
	if options.follow {
		info, statErr := os.Stat(path)
		if statErr != nil {
			return localFailure("VALIDATED", "ERROR", "logs", statErr.Error())
		}
		result.follow = &followSpec{path: path, stream: options.stream, offset: info.Size()}
	}
	return result
}

func followLog(ctx context.Context, spec *followSpec, jsonOut bool) error {
	if spec == nil {
		return nil
	}
	offset := spec.offset
	ticker := time.NewTicker(250 * time.Millisecond)
	defer ticker.Stop()
	for {
		select {
		case <-ctx.Done():
			return nil
		case <-ticker.C:
			lstat, err := os.Lstat(spec.path)
			if err != nil {
				return err
			}
			if lstat.Mode()&os.ModeSymlink != 0 {
				return errors.New("arquivo de log symlink não é aceito")
			}
			info, err := os.Stat(spec.path)
			if err != nil {
				return err
			}
			if info.Size() < offset {
				offset = 0
			}
			file, err := os.Open(spec.path)
			if err != nil {
				return err
			}
			if _, err := file.Seek(offset, io.SeekStart); err != nil {
				file.Close()
				return err
			}
			reader := bufio.NewReader(file)
			for {
				line, readErr := reader.ReadString('\n')
				if len(line) > 0 {
					offset += int64(len(line))
					line = strings.TrimSuffix(strings.TrimSuffix(line, "\n"), "\r")
					if jsonOut {
						payload := successEnvelope("LOG_LINE", "VALIDATED", map[string]any{"stream": spec.stream, "line": line})
						encoded, _ := json.Marshal(payload)
						fmt.Println(string(encoded))
					} else {
						fmt.Println(line)
					}
				}
				if errors.Is(readErr, io.EOF) {
					break
				}
				if readErr != nil {
					file.Close()
					return readErr
				}
			}
			file.Close()
		}
	}
}

func runCommand(app *App, args []string) commandResult {
	parsed, err := parseArgs(args)
	if err != nil {
		return localFailure("VALIDATED", "SYNTAX", "syntax", err.Error())
	}
	if parsed.command == "help" || parsed.command == "-h" || parsed.command == "--help" {
		if len(parsed.args) != 0 {
			return localFailure("VALIDATED", "SYNTAX", "syntax", "help não aceita argumentos")
		}
		return commandResult{response: successEnvelope("HELP", "VALIDATED", map[string]any{"usage": helpText()}), exitCode: 0, human: helpText()}
	}
	if parsed.command == "version" {
		if len(parsed.args) != 0 {
			return localFailure("VALIDATED", "SYNTAX", "syntax", "version não aceita argumentos")
		}
		return versionCommand(parsed.jsonOut)
	}
	dir, err := configDir()
	if err != nil {
		return localFailure("VALIDATED", "ERROR", "storage", err.Error())
	}
	if parsed.command == "config" {
		return configCommand(dir, parsed.args)
	}
	if parsed.command == "logs" {
		return logsCommand(app, dir, parsed.args)
	}
	if parsed.command != "start" && parsed.command != "stop" && parsed.command != "status" && parsed.command != "health" {
		return localFailure("VALIDATED", "SYNTAX", "syntax", "comando desconhecido; use help")
	}
	if len(parsed.args) != 0 {
		return localFailure("VALIDATED", "SYNTAX", "syntax", "este comando não aceita argumentos livres")
	}
	config, err := loadConfig(dir)
	if err != nil {
		return localFailure("VALIDATED", "INVALID_CONFIG", "config", err.Error())
	}
	switch parsed.command {
	case "start":
		return startCommand(app, dir, config)
	case "stop":
		return stopCommand(app, dir, config)
	case "status":
		return statusCommand(app, config)
	default:
		return healthCommand(app, dir, config)
	}
}

func printResult(result commandResult, jsonOut bool) {
	if result.raw && !jsonOut {
		data, _ := dataObject(result.response)
		if lines, ok := data["lines"].([]any); ok {
			for _, line := range lines {
				fmt.Println(line)
			}
		}
		return
	}
	if result.human != "" && !jsonOut {
		fmt.Println(result.human)
		return
	}
	encoded, err := json.Marshal(result.response)
	if err != nil {
		fmt.Printf("{\"schema_version\":1,\"ok\":false,\"code\":\"ERROR\",\"profile\":\"VALIDATED\",\"data\":{},\"error\":{\"kind\":\"json\",\"message\":%q}}\n", err.Error())
		return
	}
	fmt.Println(string(encoded))
}

func humanFor(response ManagerEnvelope) string {
	if response.Error != nil {
		return fmt.Sprintf("erro: %s", response.Error.Message)
	}
	switch response.Code {
	case "ALREADY_RUNNING":
		return "Serviço já está ativo; nenhuma alteração foi feita."
	case "STOPPED":
		return "Serviço parado com um SIGTERM verificado."
	case "NOT_RUNNING":
		return "Nenhum serviço gerenciado está ativo."
	case "PREPARED":
		return "Root isolado preparado; aguardando inicialização."
	case "READY":
		return "Serviço pronto."
	case "HEALTHY":
		return "Health check PASS."
	case "CONFIG", "CONFIG_UPDATED", "CONFIG_RESET":
		data, _ := dataObject(response)
		if key, ok := data["key"].(string); ok {
			return fmt.Sprintf("%s: %s=%v", response.Profile, key, data["value"])
		}
		return fmt.Sprintf("%s: context-length=%v max-total-tokens=%v random-seed=%v startup-timeout-seconds=%v",
			response.Profile, data["context-length"], data["max-total-tokens"], data["random-seed"], data["startup-timeout-seconds"])
	case "VERSION":
		return "qwenctl " + version
	case "HELP":
		return helpText()
	default:
		if response.OK {
			return response.Code
		}
		return fmt.Sprintf("erro: %s", response.Code)
	}
}

func main() {
	app := NewApp()
	result := runCommand(app, os.Args[1:])
	parsed, _ := parseArgs(os.Args[1:])
	jsonOut := parsed.jsonOut
	if result.human == "" && !result.raw && !jsonOut {
		result.human = humanFor(result.response)
	}
	printResult(result, jsonOut)
	if result.follow != nil {
		ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
		_ = followLog(ctx, result.follow, jsonOut)
		stop()
	}
	if result.exitCode != 0 {
		os.Exit(result.exitCode)
	}
}
