# Qwen3.8-27B NORMAL — guia da configuração final

Este é o guia curto para usar, conferir e reproduzir a melhor configuração validada no RTX 5090 do gpu-host. O histórico científico completo continua em `OPTIMIZATION.md`.

## 1. O que está funcionando agora

Não é necessário iniciar nada enquanto o serviço responder.

| Item | Valor validado |
|---|---|
| Controle | `qwenctl.exe` instalado em `%LOCALAPPDATA%\Programs\QwenCtl` |
| Root WSL ativo | `/home/operator/qwen38-5090/optimization-runs/qwenctl-20260910T012754Z-a3c64c11` |
| Template imutável | `/home/operator/qwen38-5090/optimization-runs/pi-serving-059` |
| Endpoint no Mac | `http://127.0.0.1:8002` |
| Caminho do túnel | `8002 → gpu-host:8081` |
| Modelo público | `qwen3.8-27b-normal` |
| Modelo | `/home/operator/qwen38-5090/models/Qwen3.8-27B-NVFP4-LMHead4` |
| Draft DFlash2 | `/home/operator/qwen38-5090/models/DFlash2-NVFP4-RTNcal` |
| Contexto | `131200` |
| Pool de tokens | `131328` |
| Atenção | FlashInfer, DFlash2 block 8, thresholds `1.0/1.0` |
| GEMM | FlashInfer CUTLASS, com KDA seletivo nos dois shapes validados |
| Potência | `460 W` |

## 2. Conferência mais simples

Abra um PowerShell ou Prompt de Comando novo no Windows:

```powershell
qwenctl status
qwenctl health
```

`health` confere identidade do processo, modelo, contexto, pool, DFlash, backend GEMM, root/hook KDA, os nove hashes AOT e o limite de 460 W. O resultado correto é `HEALTHY` com perfil `VALIDATED`.

Comandos principais:

```powershell
qwenctl start
qwenctl stop
qwenctl config show
qwenctl logs
```

Para apenas listar o modelo exposto pela API no Mac:

```bash
curl -s http://127.0.0.1:8002/v1/models | python3 -m json.tool
```

## 3. O que foi alterado para ganhar velocidade

O modelo e o SGLang instalado **não foram editados**.

O arquivo process-local `sitecustomize.py` amplia a lista de shapes já suportados pelo kernel KDA apenas para:

```text
(8, 17408, 5120)    # down projection
(8, 5120, 248320)   # LM head
```

Antes de fazer isso, o hook:

1. confirma que está no root exato do serviço;
2. carrega nove bibliotecas AOT privadas e verifica seus hashes;
3. verifica o SHA-256 do arquivo KDA instalado;
4. recusa iniciar se o SGLang já tiver mudado a elegibilidade desses shapes.

Portanto, uma atualização incompatível falha fechada em vez de executar silenciosamente uma configuração diferente.

## 4. Como o resultado foi provado

O candidato KDA2 venceu controles executados imediatamente antes e depois:

- controle anterior: `9/9` cenários, média geométrica `+1,7421%`;
- controle posterior: `9/9` cenários, média geométrica `+2,7542%`;
- performance: `27/27` casos com 1024 tokens de saída;
- contexto: total testado de `131110` tokens;
- needles: `3/3` respostas JSON exatas;
- matemática, raciocínio e ferramentas: PASS;
- sampling normal, seeds `17`, `29` e `43`: PASS;
- memória: estável durante 125 segundos;
- launcher Windows, túnel e processo Linux: persistentes.

A correção experimental para `min_p` positivo funcionou semanticamente, mas não foi promovida porque a combinação perdeu performance. O fast path usado normalmente continua `min_p=0`.

## 5. Onde tudo está salvo

### Fonte e operação no Mac

- `OPTIMIZATION.md` — registro append-only de hipóteses, tentativas, números e falhas;
- `FINAL_RUNBOOK.md` — este guia;
- `scripts/sitecustomize-kda059.py` — hook KDA final;
- `scripts/spawn059.ps1` — lançamento Windows original;
- `scripts/exec-manifest.py` — executor com verificações de segurança;
- `windows/qwenctl/` — repositório Git local, código Go/Python, instalador e binário Windows;
- `windows/qwenctl` — repositório privado remoto;
- `scripts/check-final059.py` — health check histórico do root template;
- `scripts/recon056.py`, `scripts/quality-api059.py`, `scripts/stochastic-quality059.py` — validações;
- `.pi/recovery/unstuck/20260909T194255Z-attention-path.md` — caderno causal encerrado como `RESOLVED`.

### Evidência principal

- `evidence/pair047048-summary.json`;
- `evidence/pair047048050-summary.json`;
- `evidence/performance048.jsonl`;
- `evidence/needles048.jsonl`;
- `evidence/quality059.jsonl`;
- `evidence/stochastic-quality059.jsonl`;
- `evidence/pi-smoke059.json`;
- `evidence/recon059.json`;
- `evidence/memory-stability059.json`;
- `evidence/final-audit059.json`.

### Bundle executável no gpu-host

`pi-serving-059` é o template validado e imutável. O `qwenctl` criou o root ativo `qwenctl-20260910T012754Z-a3c64c11`, com `launch.json`, hook, AOT e caches copiados para arquivos independentes. O estado e os logs Windows ficam em `%LOCALAPPDATA%\QwenCtl`.

## 6. Reiniciar ou criar uma réplica

### Uso normal

```powershell
qwenctl stop
qwenctl start
qwenctl health
```

O ciclo real foi validado: o app parou `pi-serving-059` com um único `SIGTERM`, confirmou `STOPPED`, criou um root novo e retornou `READY/HEALTHY`. Cada novo start usa outro root isolado; nada antigo é apagado.

### Regra importante

**Não execute novamente `spawn059.ps1`, não apague `server-pid.json` e não reutilize manualmente o root `pi-serving-059`.** O `qwenctl` faz as verificações de PID/boot/controller/fila, cópia AOT independente, root pin e readiness automaticamente. Não existe modo force.

### Replicar em outra máquina

É necessário reproduzir primeiro a plataforma exata:

- RTX 5090 e driver compatível;
- SGLang `0.5.19.dev992+g01e19ddf5`;
- FlashInfer `0.6.18`;
- Torch `2.13.0+cu130`;
- os dois modelos nos caminhos configurados;
- KDA instalado com SHA-256 `b14c3eaa6b8d9d62b559a50be62fe4cad935e7172bdb79d18a1f6abcb4ccb7c5`;
- `launch.json`, hook KDA e AOT com hashes correspondentes.

Depois, use um root isolado, altere o `expected_root`, inicie pelo controller persistente e execute todos os gates da seção 4. Sem esses gates, a réplica é apenas candidata, não a configuração validada.

## 7. O que não fazer

- não atualizar SGLang, FlashInfer, Torch ou driver sem nova validação;
- não alterar clock, potência, segurança ou permissões dos contadores NVIDIA;
- não editar modelos nem caches protegidos;
- não ativar `min_p` experimental no serviço final;
- não iniciar um segundo servidor enquanto o atual usa a GPU;
- não usar `kill -9`, apagar roots ou limpar caches como tentativa de recuperação.
