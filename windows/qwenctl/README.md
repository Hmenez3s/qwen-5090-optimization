# qwenctl

`qwenctl.exe` é um CLI Windows para ligar, desligar, configurar e inspecionar
o serviço Qwen3.8-27B NORMAL no Ubuntu WSL da RTX 5090 do gpu-host. Ele usa o
root validado `pi-serving-059` como referência imutável e, quando precisa
iniciar, prepara um root `qwenctl-...` novo e isolado.

O perfil padrão venceu controles contemporâneos em 9/9 cenários e mediu
**303,14 tok/s** no caso curto de código. Esse número é um resultado medido,
não uma garantia para qualquer prompt: contexto, tipo de saída e raciocínio
alteram a velocidade.

## Instalação

No Windows, a partir desta pasta, execute no PowerShell 5.1:

```powershell
.\Install-QwenCtl.ps1
```

A instalação é somente por usuário em `%LOCALAPPDATA%\Programs\QwenCtl`, não
requer administrador e não cria serviço, tarefa agendada ou inicialização
automática. O arquivo `%LOCALAPPDATA%\QwenCtl\config.json`, se existir, é
preservado. O instalador controla uma instalação SGLang/modelos já existente;
ele não baixa os pesos do modelo nem altera drivers ou pacotes.

Abra um novo terminal depois de alterar o PATH. Comece por:

```text
qwenctl version
qwenctl status
qwenctl health
qwenctl start
```

## Comandos

```text
qwenctl start [--json]
qwenctl stop [--json]
qwenctl status [--json]
qwenctl health [--json]
qwenctl config show
qwenctl config get <chave>
qwenctl config set <chave> <valor>
qwenctl config reset
qwenctl logs [--lines N] [--stdout|--stderr] [--follow]
```

Os valores permitidos são `context-length` (131072–131200),
`max-total-tokens` (de `context-length` até 131328), `random-seed`
(0–2147483647) e `startup-timeout-seconds` (60–1800). Os padrões validados
são 131200, 131328, 7319 e 900. Qualquer alteração aceita aparece como
`CUSTOM_UNVALIDATED` e só vale no próximo `start`.

`start` não inicia um segundo serviço se o serviço validado ou um root gerido
já estiver ativo. `stop` só envia um SIGTERM depois de confirmar root, modelo,
PID/boot/start identity, exatamente um controller Windows e métricas de fila e
execução em zero. Não existe modo force: não há SIGKILL, limpeza de root ou
remoção de logs.

`--json` produz um envelope estável com `schema_version`, `ok`, `code`,
`profile`, `data` e `error`. Mensagens interativas e erros são em português.
Os logs são os arquivos redirecionados pelo controller persistido localmente;
o padrão é stderr e 100 linhas.

## Build local

No diretório deste arquivo:

```text
make test
make build
make package
```

O build é CGO-free para Windows AMD64. `dist/` contém somente artefatos
regeneráveis; o código-fonte não depende de bibliotecas de terceiros.
