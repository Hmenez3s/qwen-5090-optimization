# Caderno de laboratório — Qwen3.8-27B na RTX 5090

## Natureza e disciplina do caderno

Este arquivo é o **caderno de anotações científicas da tarefa**, não apenas um plano ou checklist. É a referência única para hipóteses, procedimentos, evidências, resultados negativos, decisões e retomada. Logs e resultados brutos ficam como anexos referenciados; não são um segundo registro de controle.

- O orquestrador é o único escritor; agentes retornam evidência para incorporação.
- Resumo do estado e pendências podem ser atualizados. O histórico de experimentos é **append-only**: correções entram como adendos ligados ao ID original, sem apagar resultados ou reescrever previsões.
- Antes de executar, registrar hipótese, previsão falsificável, variáveis/controlos, comando/configuração, critério de decisão e limite de execução. Finalizar o resultado antes de iniciar o próximo experimento.
- Tentativas anteriores à criação do caderno são registros **retrospectivos**, nunca fingir que foram pré-registradas.
- Uma falha, ausência de ganho ou hipótese refutada tem o mesmo valor documental de um resultado positivo. Registrar também experimentos não executados e o motivo.
- Cada entrada nova deve ter data/hora UTC, versão/identidade dos artefatos e caminhos das evidências. Não inventar datas ou hashes ausentes nas entradas retrospectivas.
- Não registrar credenciais, conteúdo privado de conversas ou dumps sensíveis.

### Vocabulário de evidência

| Rótulo | Uso |
|---|---|
| `OBSERVED` | Observado diretamente nesta execução em resposta, log, arquivo ou medição. |
| `RETRIEVED` | Alegação recuperada de fonte externa ou histórico, com proveniência; não reproduzida aqui. |
| `INFERRED` | Conclusão derivada de evidências citadas, explicitando a derivação. |
| `MODEL_PRIOR` | Intuição/conhecimento prévio usado para propor hipótese, não comprovação. |
| `UNKNOWN` | Questão material ainda sem resposta. |
| `REFUTED` | Alegação contradita por evidência identificada. |

Hipóteses: `OPEN | SUPPORTED | WEAKENED | REFUTED | UNRESOLVED`. Resultados de um modelo, tarefa ou contexto não são extrapolados automaticamente para outros.

## Objetivo e critérios

Maximizar primeiro a velocidade de geração **por conversa do Qwen3.8-27B normal**, sem teto predefinido de tokens/s, na RTX 5090 32 GB do gpu-host. Manter contexto utilizável ≥131.072 tokens, qualidade, raciocínio e ferramentas. 200 tok/s não é teto nem critério obrigatório de encerramento. **Abliterated é uma segunda etapa opcional e condicional a obter excelente performance no normal; não é requisito nem prioridade da primeira etapa.**

Preservar:
- Contexto utilizável de pelo menos **131.072 tokens**, compartilhado entre entrada, raciocínio e saída.
- Qualidade, coerência, raciocínio e chamadas de ferramentas; não promover aceleração que altere indevidamente a geração.
- Modelos existentes, dados, SSH e acesso pelo túnel `8002 → 8081`.

Comparar SGLang, NInfer Turbo e drafts compatíveis, incluindo DFlash2, DSpark e MTP quando suportados pelo checkpoint/runtime. Priorizar configuração e kernels nativos existentes; modificar kernels somente com gargalo identificado por profiling. Não treinar outro modelo ou draft por padrão.

Velocidade individual e throughput agregado são métricas separadas. Medir contexto curto, intermediário e próximo do limite, reservando espaço para saída. Não confundir contexto configurado, capacidade de alocação e contexto realmente exercitado com qualidade.

## Autonomia e autorização

O usuário pediu execução autônoma, sem perguntas rotineiras ou confirmação a cada ajuste técnico. Escolhas reversíveis dentro do escopo devem ser resolvidas por evidência.

**Situação atual:** janela explicitamente autorizada pelo usuário na ativação do goal `c2be2a18-a329-4e73-b538-16dabebef5af`: pausas/reinícios e trocas reversíveis de configuração/checkpoint do servidor de inferência, indisponibilidade temporária e perda do cache em memória. Aprovação é a mensagem do usuário, não este arquivo. Incidentes suspendem mutações e não autorizam recuperação automática. Ver adendo A01 para o estado anterior.

Sem autorização para apagar dados/modelos existentes, alterar segurança/SSH/firewall, fazer compras, publicar, ou realizar overclock. Incidentes e ações fora do escopo exigem parada e comunicação. Não repetir testes em estado compartilhado sem resolver isolamento e propriedade dos recursos.

## Estado atual

- `OBSERVED`: PI028 retornou READY após manutenção MICRO027 (54s de startup), alias `qwen3.8-27b-normal` em8081 e PI via8002 reconfirmados. Perfil de referência016, sem hook KDA experimental. Evidências `spawn028.txt` e resultados MICRO027; não extrapolar para estabilidade futura.
- `OBSERVED`: contexto lógico131200/pool131328; ≥131072 utilizáveis exercitados nos experimentos anteriores. Raciocínio/tool roundtrip passaram pelo adapterPI; qualidade estocástica ampla continua `UNKNOWN`.
- Goal ativo: `3c0032b7-57bc-45d3-bf5c-0904f9ad6266`, explicitamente retomado para investigação profunda orientada aestaRTX5090. Manutenção autorizada, mas nenhuma medição na KV/cache de conversas reais; testes GPU somente em processo/cenário isolado.
- `OBSERVED`: KDA no microteste~47% mais rápido que CUTLASS default; E2E023 apenas+1,06% provisório, controle contemporâneo interrompido. Não extrapolar e não declarar teto de ganho.
- Próximo: igualar seleção/tuning dos GEMMs ao servidor; instrumentação de memória/compute/esperas. `UNKNOWN`: disponibilidade real dos contadores deperformance sobWSL. Nenhuma alteração de segurança, clocks, potência, SSH ou modelos.
- Normal primeiro; abliterated adiado. Histórico abaixo permanece append-only; o resumo acima substitui somente estado transitório anterior.

## Ambiente observado por SSH

| Item | Evidência observada |
|---|---|
| Host | `operator@192.0.2.12` |
| SO | Windows 11 Pro, Ubuntu em WSL2 |
| GPU | NVIDIA GeForce RTX 5090, 32.607 MiB reportados |
| Driver | `610.88` |
| Limite de potência observado | 460 W; não alterado |
| VRAM em uma leitura | 31.063 MiB usados, 1.125 MiB livres |
| SGLang | `0.5.19.dev992+g01e19ddf5` |
| Commit instalado | `01e19ddf55ad93b7e7459a03aeee43b3892bb567` |
| Ambiente Python | `/home/operator/.venvs/qwen38` |
| API no WSL | `http://127.0.0.1:8081/v1` |
| Modelo servido | `qwen38` |
| API pelo túnel no Mac | `http://127.0.0.1:8002/v1` |

Todos os dados de processo, memória e disponibilidade são snapshots; revalidar antes de operar. Não usar PID antigo como identidade suficiente.

Túnel informado pelo usuário:

```bash
ssh -N -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 -L 127.0.0.1:8002:127.0.0.1:8081 operator@192.0.2.12
```

## Configuração ativa observada

- Target: `/home/operator/qwen38-5090/models/Qwen3.8-27B-NVFP4-LMHead4`.
- Draft: `/home/operator/qwen38-5090/models/DFlash2-NVFP4-RTNcal`.
- Quantização: `modelopt_fp4` nos dois modelos; GEMM `flashinfer_cutlass`.
- Especulação: DFLASH, bloco 8, 8 draft tokens, janela do draft 2048.
- Attention: `flashinfer`; KV: `fp8_e4m3`.
- Contexto e pool: `131072`; `max_req_input_len` observado: `131066`.
- Uma requisição por vez; single-batch overlap.
- Prefill: 4096; `mem_fraction_static=0.92`.
- Estados GDN/Mamba: BF16, cinco slots.
- Parsers: `qwen3` e `qwen3_coder`.
- Somente linguagem via `json_model_override_args`.
- Graphs de target verify e draft decode capturados, confirmados pelas durações de startup em `/server_info`.
- Prefill CUDA Graph desativado; métricas Prometheus não habilitadas (`/metrics` retornou 404).
- HiCache, caches externos e sessões persistentes não habilitados nas flags observadas. Isso não torna o serviço compartilhado descartável.

## Autoridade de inicialização e recuperação

Diretório operacional existente no Windows:

```text
C:\Users\operator\Documents\QwenResearch\historical\outputs\qwen38-windows
```

Correspondente no WSL:

```text
/mnt/c/Users/operator/Documents/QwenResearch/historical/outputs/qwen38-windows
```

Cadeia observada: `start-shared-131k.ps1` → `start-shared-lmhead4-131k.ps1` → `start-server.sh` no mesmo diretório. Esta é a cadeia a reconciliar/manter; não criar um segundo mecanismo permanente de inicialização.

**Não confundir** com `/home/operator/qwen38-5090/start-server.sh`: seus defaults apontam para outro perfil/modelo/porta/contexto e não reproduzem o servidor atual.

O perfil antigo `start-shared-radixark-131k.ps1` existe, mas a auditoria registra pouca folga de VRAM. Não é rollback recomendado automaticamente. Preservar a configuração efetivamente ativa e conferir dependências antes de qualquer manutenção. O histórico diz não haver autostart no Windows; revalidar se necessário.

## Medições desta sessão — baseline preliminar

Executado anteriormente na conversa, antes da etapa atual de otimização:

```bash
/home/operator/.venvs/qwen38/bin/python /home/operator/qwen38-5090/benchmark.py --url http://127.0.0.1:8081/v1 --max-tokens 512 --rounds 1
```

| Prompt curto | Tokens | TTFT | E2E tok/s | Streaming aproximado tok/s |
|---|---:|---:|---:|---:|
| Implementação LRU Python | 512 | 133 ms | 189,7 | 199,5 |
| Desconto e reajuste | 266 | 134 ms | 260,1 | 299,3 |
| Explicação de especulação | 512 | 120 ms | 129,3 | 133,3 |
| Mediana dos três | — | 133 ms | 189,7 | 199,5 |

Limitações: uma rodada, contextos curtos, servidor compartilhado, respostas parcialmente limitadas pelo orçamento. A métrica `stream_tps` divide todos os tokens pelo intervalo a partir do primeiro texto; não desconta os tokens do primeiro chunk e pode superestimar decode. Não serve como benchmark isolado, prova de qualidade ou resultado com 131k preenchidos. A execução passada não estabelece isolamento para novos testes.

## Evidência histórica encontrada — não são novas medições

Fonte: `AUDITORIA-PERFORMANCE.md` e JSONs no diretório operacional, datados no documento como 2026-09-08.

| Arquivo | Entrada | Saída | Decode tok/s | E2E tok/s | Observação |
|---|---:|---:|---:|---:|---|
| `audit-lmhead4-short.json` | 19 | 512 | 364,18 | 308,70 | Tarefa curta específica |
| `audit-lmhead4-131k.json` | 130048 | 512 | 202,19 | 14,61 | TTFT 32,522 s |
| `audit-lmhead4-131k-repeat.json` | 130048 | 512 | 210,19 | 131,01 | 126976 tokens em cache |
| `audit-final-16k-sampling.json` | 16384 | 1024 | 267,14 | 181,64 | Temperatura 1 |
| `audit-final-131k.json` | 130048 | 512 | 196,92 | 12,97 | 12288 tokens em cache; outra requisição em fila |

O histórico usa `(tokens finais − tokens do primeiro bloco) / intervalo entre blocos` para decode. Contexto longo formado por código do SGLang para uma tarefa LRU; não prova recuperação de informação, qualidade geral ou 131072 tokens exatos de sequência preenchida. Resultados históricos pertencem aos respectivos perfis; não misturar variantes ou atribuir ganhos desta tarefa.

## Lições verificadas no histórico

- Pressão de VRAM já impediu captura do graph do draft e acompanhou forte queda de velocidade. Confirmar captura real, não apenas flags.
- Experimentos NGRAM/ReplaySSM chegaram a 400+ tok/s, mas uma promoção agressiva de estado mudou os tokens. Não promover esses números como aceleração equivalente.
- Fused KV materialization do DFlash foi mantida desativada por incompatibilidade com QKV quantizado no código então instalado. Verificar suporte antes de tentar habilitar.
- README e pesquisa mais antigos divergem da auditoria final sobre o target ativo. Priorizar evidência atual do processo e a cadeia de inicialização correspondente.
- Um teste anterior de seleção de Python via shell/PowerShell falhou antes de iniciar inferência; usar o caminho conhecido do ambiente e transporte de argumentos sem expansão acidental.

## Protocolo de experimentos

1. Revalidar identidade dos pesos, versão/runtime, controle do processo, autorização e recuperação.
2. Provar isolamento de recursos antes de iniciar benchmark/servidor experimental: processo e caches próprios, diretório temporário próprio, pesos existentes somente leitura, prompts sintéticos, nenhuma ferramenta real ou serviço externo, sem fallback para API compartilhada. Não apagar/limpar caches existentes.
3. Estabelecer baseline repetido nas mesmas tarefas, templates, sampling, limites e condições de memória. Separar cache frio/quente e tráfego concorrente.
4. Alterar uma variável por comparação: draft, bloco suportado, prefill, backend/graphs. Não comparar modelos diferentes como se fossem apenas otimizações de runtime.
5. Registrar velocidade individual, E2E, TTFT, aceitação, VRAM, contexto efetivamente consumido, correção e dispersão entre repetições.
6. Promover somente ganhos acima do ruído com critérios preservados; repetir baseline para detectar drift.
7. Encerrar ao esgotar hipóteses fundamentadas ou quando o ganho ficar no ruído. Deixar a melhor configuração validada funcionando, se a operação tiver sido autorizada. Não prometer execução infinita ou trabalho em background inexistente.

## Registro de execução

| ID | Ação/hipótese | Resultado | Decisão |
|---|---|---|---|
| R01 | Inventário read-only de GPU, WSL e servidor | Instalação SGLang existente, GPU ocupada | Reutilizar conhecimento; não instalar do zero |
| R02 | Consultar métricas | `/metrics` 404; `/server_info` disponível | Metadados úteis, sem Prometheus atual |
| B01 | Três prompts curtos, uma rodada | Mediana E2E 189,7; streaming aprox. 199,5 tok/s | Baseline preliminar, não isolado |
| R03 | Ler model cards, launchers e histórico | Target não documentado como abliterated; graphs ativos; histórico 131k localizado | Separar alvo final de perfil atual |
| R04 | Reconciliação independente read-only do snapshot | Recuperação e limitações de benchmark identificadas | Não repetir experiências inválidas |
| D01 | Criar este registro dedicado | Tracker local criado; sem alteração no host | Atualizar aqui a cada ciclo |

As entradas R01–D01 acima são retrospectivas. Medições de arquivos históricos permanecem identificadas como `RETRIEVED`, e não como resultados reproduzidos pelo trabalho atual.

### Modelo de entrada científica — preencher antes e depois

```markdown
### EXP-NNN — título
Tipo: pré-registrado | retrospectivo
Estado: planejado | em execução | concluído | não executado
Registro UTC / início UTC / fim UTC:

#### Antes da execução
- Pergunta e decisão que o resultado mudará:
- Hipótese H-NNN e estado inicial:
- Evidências de origem e seus rótulos:
- Explicação concorrente e como distingui-la:
- Previsão falsificável; observação que refutaria a hipótese:
- Variável independente (única mudança/coerente) e controles:
- Target/draft: origem, revisão, quantização, template:
- Runtime: commit, dependências, kernels/graphs, GPU/driver:
- Configuração completa baseline → candidata e comandos exatos:
- Recursos atingidos, isolamento, autorização e recuperação:
- Dataset/prompts sintéticos, tokens, output budget, sampling/seed:
- Warmup, cache frio/quente, concorrência, repetições e ordem A/B:
- Métricas, fórmula, dispersão e limiar de ganho definidos previamente:
- Oráculos de qualidade, contexto, raciocínio e ferramentas:
- Limites de tempo/carga/saída e condição de parada:

#### Depois da execução
- Execução real e desvios do procedimento:
- Evidência bruta: caminho, hash quando disponível, saída/erro:
- OBSERVED: TTFT, decode, E2E, aceitação, VRAM, tokens reais:
- Estatística: amostra, mediana, dispersão, efeito absoluto/relativo:
- Correção/qualidade/contexto: passou | falhou | não verificado:
- Confundidores e limitações:
- Previsão versus resultado; hipótese atualizada:
- Veredito: manter | rejeitar | inconclusivo | não executado:
- Recursos protegidos e encerramento do experimento verificados:
- Próxima pergunta e evidência nova necessária:

#### Adendos posteriores (sem reescrever acima)
- UTC / correção / evidência / conclusão afetada:
```

### Registro anti-repetição

Uma nova tentativa precisa produzir informação que mude uma decisão. Replicações estatísticas são permitidas quando fazem parte do protocolo pré-registrado; insistência após falha sem nova evidência não é replicação válida.

| Caminho | Evidência/estado | Condição para reconsiderar |
|---|---|---|
| NGRAM/ReplaySSM com promoção agressiva e números 400+ | `RETRIEVED`: histórico relata divergência de tokens | Nova correção identificada e oráculo de equivalência; não repetir a mesma configuração |
| Habilitar fused KV com QKV quantizado sem checar suporte | `RETRIEVED`: limitação documentada no runtime anterior | Evidência versionada de suporte ou correção isolada verificável |
| Aumentar pool até esgotar VRAM | `RETRIEVED`: graph do draft já foi desativado por pressão de memória | Orçamento de memória e captura reais, incluindo prefill e uso externo |

### Escalonamento com `unstuck`

O usuário pediu usar `unstuck` quando o problema ficar especialmente difícil ou parecer que não há como otimizar mais. Skill localizada em `/Users/operator/.pi/agent/skills/unstuck/SKILL.md` e lida.

**Limite da skill atualmente instalada:** exige invocação explícita `/skill:unstuck`; proíbe autoativação e sugestão pelo modelo. O pedido de uso futuro fica registrado, mas não será convertido em alegação de que a skill já está ativa nem em alteração de suas regras. Quando houver invocação válida, carregar a versão vigente e seguir o procedimento, sem ampliar autorização ou relaxar segurança.

O método do caderno já exige hipóteses concorrentes, previsões falsificáveis e evidência nova para evitar loops. Em impasse, registrar exatamente o que se sabe, o que foi refutado e o que falta; não inventar ganho nem executar indefinidamente.

A skill exige um caderno único por invocação. Se ativada, manter esse artefato como **anexo científico delimitado** sob esta pasta da tarefa, referenciado aqui; `OPTIMIZATION.md` continua sendo o índice e estado geral autoritativos, sem duplicar controle de experimentos.

## Pendências

- [x] Reconciliar a autorização da janela de manutenção do serviço compartilhado; grant explícito na ativação do goal.
- [ ] Otimizar e verificar primeiro o Qwen normal; avaliar abliterated apenas depois, como segunda etapa opcional.
- [ ] Preparar execução experimental com isolamento comprovado e recuperação.
- [ ] Medir baseline repetível curto/intermediário/longo e qualidade.
- [ ] Comparar drafts/runtimes e ajustar com uma hipótese por vez.
- [ ] Aplicar e verificar melhor perfil autorizado; atualizar inicialização existente sem mecanismo concorrente.
- [ ] Registrar resultado final e limites, sem extrapolar benchmarks.

## Referências

- SGLang cookbook: https://docs.sglang.io/cookbook/autoregressive/Qwen/Qwen3.8-27B
- Abliterated NVFP4 candidato: https://huggingface.co/heswithme/Huihui-Qwen3.8-27B-Abliterated-Gittensor-Style-NVFP4-RTX5090
- DSpark: https://huggingface.co/RadixArk/Qwen3.8-27B-DSpark
- DSpark NVFP4: https://huggingface.co/gittensor-model-hub/Qwen3.8-27B-DSpark-NVFP4
- DFlash2: https://huggingface.co/incoai/Qwen3.8-27B-DFlash2
- NInfer, metodologia fornecida pelo usuário, ainda não verificada nesta tarefa: https://github.com/Neroued/ninfer/blob/master/docs/performance.md
- Receita RTX 3090 fornecida pelo usuário, não diretamente equivalente à 5090: https://github.com/syv-ai/qwen38-27b-rtx3090

O post de NInfer informado pelo usuário usa oito streams, 12288 tokens de saída por stream e pool KV compartilhado de 187712 tokens. Não converter esses dados em velocidade individual nem capacidade de 131k por requisição sem evidência adicional.

## A01 — ativação e reconciliação de autoridade — 2026-09-08T20:36:42.432790+00:00

Antes da ativação, a manutenção e o goal estavam pendentes: o texto era somente rascunho do assistente. A mensagem atual do usuário ativou o goal e autorizou expressamente a janela descrita acima. Nenhuma aprovação histórica foi inventada; os resumos atuais foram atualizados por este adendo.

### EXP-001 — revalidar ambiente e capturar recuperação
Tipo: pré-registrado. Estado: planejado. Registro UTC: 2026-09-08T20:36:42.432790+00:00.
- Decisão: reutilizar instalação/artefatos atuais ou reorientar se houve mudança.
- H001 `OPEN`: mesmo alvo stock/DFLASH e runtime encontrados anteriormente; alternativa: outro controlador alterou processo/pesos. Refutação: identidade/config atual divergir.
- Procedimento: SSH BatchMode/StrictHostKeyChecking para gpu-host; ler /server_info somente campos não sensíveis, /proc do processo servidor, GPU, discos e launchers conhecidos. Copiar scripts/configs e metadados públicos para anexos locais, com SHA256; nenhuma leitura de credenciais ou prompts reais.
- Recursos: consulta read-only ao Windows/WSL e API metadata; não iniciar inferência, limpar cache, iniciar/parar processo ou importar runtime GPU. Escritas somente em anexos novos desta pasta local.
- Limites: SSH 60s por chamada; HTTP 10s; arquivos enumerados ≤200KB cada; sem varredura indiscriminada.
- Métricas: identidade, versão, memória, PID/starttime, suporte de kernels e launchers; sem alegação de velocidade/qualidade.
- Comandos serão preservados no anexo scripts/remote.py e no probe recon-001.py.

#### EXP-001 — resultado 2026-09-08T20:37:50.541708+00:00
`OBSERVED`: transporte recusado com `The command line is too long.`; probe não executou. Resultado de identidade `UNRESOLVED`. Nenhum benchmark/mutação remota.

### EXP-001B — transporte compacto
Pré-registro: comprimir o Python com zlib antes da codificação PowerShell, preservando o mesmo probe read-only. Previsão: comando abaixo do limite Windows e JSON de metadados retornado; erro do mesmo limite refuta a solução. Mesmo escopo/limites de EXP-001. Arquivo anterior vazio preservado; saída em recon-001b.json.

#### EXP-001B — resultado 2026-09-08T20:38:28.714238+00:00
`OBSERVED`: mesmo processo 834, target LMHead4/DFLASH e SGLang `01e19ddf5`. Graphs de verify/draft capturados; 1475 MiB VRAM livres, disco WSL 847 GiB. H001 `SUPPORTED`. Snapshot em evidence/recon-001b.json, launchers/configs com hashes. Sem inferência nem alteração remota.

### EXP-002 — proveniência do candidato e contrato NInfer
Pré-registro: obter HF API/revisão e PROVENANCE do heswithme abliterated, metadata dos drafts e docs oficiais NInfer. H002: abliterated export tem linhagem explícita e layout ModelOpt compatível; H003: NInfer aceita essa família mas pode exigir conversão/preservação MTP. Previsões refutáveis por metadados de origem/formatos e docs. Consultas HTTPS públicas, nenhuma execução de código remoto, download de pesos ou credenciais; gravar somente anexos locais. Limites 20s/requisição, texto/metadata ≤1MB. Decisão: escolher pesos e caminho de comparação, sem fingir benchmark ainda.

#### EXP-002 — resultado de aquisição
`RETRIEVED`: heswithme pin `41c5fdfad0cbe0c784981b3e44d49c164a9bd893` declara fonte Huihui BF16 `739e3c5b89849f6c238ce1e5b70008612ae42cdd`, quantização ModelOpt, dois shards ~17,92GB e MTP removido. HF API pinned fornece tamanho e SHA256 de ambos. Card da fonte recuperado na revisão declarada. Proveniência documental rastreável, não reexecução da abliteration. NInfer pin `b88c0f6fc7e999f13eb2fcf7fc9105ed79a91868` docs obtidas; `docs/getting-started.md` foi 404, nenhuma suposição de caminho será usada; scout lê referências reais. Sem pesos executados.

### EXP-003 — adquirir checkpoint abliterated fixado
Tipo pré-registrado. H004 `OPEN`: os dois shards disponíveis correspondem aos SHA256 do HF pinned. Previsão: todos os tamanhos e hashes conferem e índice aponta somente arquivos baixados; qualquer divergência bloqueia uso.
- Ação autorizada: baixar checkpoint público existente (não treinar/nem abliterar novamente) em diretório novo `models/Huihui-Abliterated-Gittensor-NVFP4-41c5fdfa`; nenhum overwrite/delete.
- Destino operacional: `/home/operator/qwen38-5090/optimization-runs/goal-c2be2a18-20260908T2042Z` para logs/manifesto de processo run-owned, não launcher permanente.
- Fontes: HF repo/revision acima; manifesto `evidence/upstream/abliterated-pinned-blobs.json`. Script `scripts/acquire-003.py` e `scripts/launch-acquire-003.py` preservam comandos.
- Recursos: HTTPS público e novos arquivos (~18GB), sem GPU/inferência/cache HF/credenciais, sem modificação dos modelos existentes. 847GiB livres observados. Arquivos abertos em modo exclusivo; nenhum fallback de modelo nem cleanup.
- Limites: 3 horas totais, 120s timeout de rede por leitura; no máximo dois shards simultâneos. Nenhuma repetição automática em falha de hash. Receipts de status/bytes/hashes no diretório novo.
- Recuperação: não há efeito sobre serviço/pesos originais; deixar arquivos novos preservados se falhar e reportar. Não promover apenas por sucesso de download.

#### EXP-003 — transporte não executado
`OBSERVED`: payload de manifesto/script novamente excedeu limite de command line Windows antes da execução remota. Compressão só resolveu probes pequenos; nova evidência invalida esse transporte como solução geral. Sem diretório remoto criado.

### EXP-003A — transporte por stdin
Pré-registro: eliminar EncodedCommand/payload dos argumentos e enviar Python por stdin a `ssh … "wsl.exe -d Ubuntu -- python3 -"`. Controle primeiro: imprimir marcador e versão, sem escrita remota. Previsão: stdin chega ao Python intacto e comando permanece constante independente do tamanho. Se passar, executar EXATAMENTE o manifesto EXP-003, sem outro alvo/método de aquisição. Limite 60s para lançamento.

## A02 — preferência de escalonamento — 2026-09-08T20:44:20.167155+00:00
O usuário solicitou: “goal edit usa a skill unstuck sempre que precisar”. Preferência registrada. A skill instalada mantém a exigência de invocação explícita `/skill:unstuck` e proíbe autoativação; não houve alteração da skill nem alegação de edição do estado runtime do goal. Continuar a investigação autônoma com hipóteses concorrentes e novas evidências, respeitando esse limite de ativação.

## A03 — correção explícita de prioridade — 2026-09-08T20:46:18.691571+00:00

O usuário corrigiu o objetivo: **não começar por abliterated**. Primeiro obter a melhor performance do **Qwen normal**; abliterated fica como segundo plano condicional. Esta decisão substitui a prioridade abliterated dos rascunhos/goal anteriores. O orquestrador reconheceu a priorização errada. A janela de manutenção e demais limites permanecem inalterados. Não alegar que o estado interno do comando /goal foi editado por ferramenta; esta é a alteração de escopo explicitamente dada pelo usuário.

EXP-003/003A: aquisição abliterated passa a `INTERROMPER_POR_MUDANÇA_DE_ESCOPO`, não falha de hipótese/qualidade. Parar somente o processo de download criado pela tarefa após conferir PID, starttime e comando contra download-pid.json; preservar todos os arquivos parciais e log, sem apagar/reiniciar nada. Nenhuma inferência abliterated ocorreu. Benchmarks e estudo NInfer passam a mirar o Qwen normal.

#### EXP-003/003A — encerramento por mudança de escopo
`OBSERVED`: downloader PID1658 validado contra starttime e comando do manifesto; SIGTERM enviado somente a ele. Pós-checagem confirmou ausência do processo. Arquivos parciais preservados; nenhum modelo original/servidor/túnel alterado. Veredito: não executar etapa abliterated agora.

## A04 — goal normal-first ativado — 2026-09-08T20:48:44.345340+00:00
Usuário ativou `065dc7bf-72b2-413a-9134-399ffbcece27` com Qwen NORMAL prioritário. Grant e limites permanecem, sem retomada do download secundário. Resultados abliterated não são baseline desta etapa.

### EXP-004 — readiness de isolamento e NInfer normal
Pré-registro: confirmar serviço/identidade atual, ler contrato de caches/lifecycle instalado e docs NInfer já nomeadas. H005: SGLang pode executar offline em processo/diretórios novos usando os pesos stock existentes somente leitura, sem atingir caches externos. H006: NInfer exige outro export; se confirmado, comparar como troca de runtime+quantização, não ganho puro de runtime. Refutação: cache/serviço externo não isolável ou contrato de pesos diferente sem conversão.
Consultas somente read-only no host, metadata HTTP e código instalado; downloads de docs públicas pin NInfer `b88c0f6fc7e999f13eb2fcf7fc9105ed79a91868`. Gravar anexos locais novos. Não iniciar modelo/importar CUDA/parar serviço. Timeouts SSH60s/HTTP20s, arquivos nomeados ≤200KB; depois decidir manifesto de primeira manutenção/benchmark.

## A05 — goal compacto ativado — 2026-09-08T20:55:03.791550+00:00
Goal atual `fb056c05-362d-4c88-b177-a7a5152b68e8`. Mesmos limites e alvo normal; platô das opções simples não encerra investigação fundamentada de profiling/CUDA/Triton.

#### EXP-004 — resultado da aquisição de evidência
`OBSERVED`: servidor continua PID834/perfil stock. CUDA toolkit existente é 13.0; NInfer exige ≥13.1. CMake/pkg-config não encontrados no PATH; ninja existe no venv. Nenhuma build/inferência executada. `RETRIEVED`: NInfer export fixado `11dbbbbbc33db198afe2f02c9232c771ff7031be`, base Qwen normal e quantização Unsloth; inclui companion DFlash2 e MTP. SHA256 publicado `552c374c685dce302603b95fbe940fb04243c0cd44c083efc644ad3d980d462c`. H006 `SUPPORTED`: comparação será runtime+export, não mesmo conjunto de pesos. H005 permanece `UNRESOLVED`, análise do código de cache/lifecycle delegada read-only; início de benchmark bloqueado até prova. Caminhos maintainer inicialmente consultados na raiz deram404; links reais são relativos a docs/, recuperados com sucesso.

### EXP-005 — aquisição do artefato NInfer normal
Pré-registro: baixar somente `qwen3_8_27b_nvfp4.ninfer` (23.719.496.192 bytes) e manifestos da revisão HF acima em diretório novo. H007: conteúdo corresponde exatamente ao SHA256 fixado; divergência bloqueia uso e não gera retry/cleanup. Não carregar modelo nem usar GPU.
- Fonte: https://huggingface.co/neroued/Qwen3.8-27B-nvfp4-NInfer/resolve/11dbbbbbc33db198afe2f02c9232c771ff7031be/ .
- Destino: `/home/operator/qwen38-5090/models/NInfer-Normal-11dbbbbb`, exclusivo; logs/manifesto em `/home/operator/qwen38-5090/optimization-runs/normal-goal-fb056c05-20260908`.
- Isolamento: novo destino/arquivos abertos exclusivos, ambiente sem credenciais nem caches globais; modelos existentes e download abliterated intocados. HTTPS público é única dependência externa.
- Limites: 3h, 120s por leitura; mínimo40GiB livres antes; duas transferências no máximo. Nenhuma exclusão. Script preservado em scripts/acquire-ninfer-005.py.
- Uso futuro condicionado a build compatível, verificação de hashes e recursos de execução isolados.

## Preparação operacional OPS-006 — 2026-09-08T21:00:12.195254+00:00
Não é um novo benchmark enquanto a aquisição EXP-005 está em andamento. Preparar baseline privado sem medir performance até cessar IO de downloads/builds concorrentes. Cliente novo está em revisão: testes offline iniciais passaram, mas revisão do orquestrador encontrou finish_reason nativo como objeto e estatísticas spec opcionais; reparo local delegado, nenhum resultado real produzido por ele.

Readiness reconciliada: target/draft locais, loader safetensors usa leitura local sem download; `delete_ckpt_after_loading=false`, sem conversão/exports/LoRA/hooks/tool-server/HiCache/externalKV. Home, cwd, armazenamento, caches SGLang/FlashInfer/Triton/CUDA/Torch e arquivos temporários serão novos e exclusivos. Ambiente construído sem credenciais/proxies, HF offline, sem trust_remote_code, arquivos Python sem bytecode. Modelos/venv existentes são entradas somente leitura; não executar o launcher antigo que pode criar symlinks no venv.

Manifesto de manutenção: conferir PID834 + starttime + argv e descendentes; preservar configuração ativa/launchers/metadata de pesos em `sglang-baseline/recovery-manifest.json` e anexo local. Porta experimental34731 somente loopback, nome servido único `qwen-normal-exp006`. Método de parada suportado pelo código instalado: SIGTERM no processo principal; TokenizerManager drena e encerra filhos; kill_process_tree com include_parent=False consulta psutil children(recursive=True), não ancestrais/SSH. A aprovação explícita cobre indisponibilidade e perda desse cache em RAM; nenhum arquivo será apagado. Estado inesperado fora do processo experimental congela mutações.

Arquivos operacionais autorizados: scripts/prepare-baseline-006.py; futuro launch-baseline-006.py; diretório novo `/home/operator/qwen38-5090/optimization-runs/normal-goal-fb056c05-20260908/sglang-baseline`. Preparação deve verificar porta livre, mesmo target e ausência de variáveis runtime não reconciliadas. Falha deixa serviço intocado; não parar antes desses gates.

## INCIDENT-006 — lifecycle WSL inesperado — 2026-09-08T21:04:55.890975+00:00

`OBSERVED`: SIGTERM foi enviado somente ao PID834 com starttime/argv revalidados. GPU liberou memória e listeners de inferência sumiram. Depois o launcher experimental criou PID400, mas esse processo desapareceu antes de readiness; log terminou após detecção de template, sem erro Python conclusivo. Journal prova transição de boot: `7cb809fdcd6c415f98c043276fd0f374` encerrou entradas18:01:00; boot `cf9e9397a9b0433386d84e672e783c80` iniciou18:02:06 (-03). Downloader NInfer PID1743 também desapareceu, deixando3.753.902.080 bytes parciais sem resultado final.

`INFERRED`: o launcher desacoplado dentro do WSL não manteve o lifetime esperado ao encerrar a sessão/controlador Windows. Causa exata entre lifecycle WSL e associação da sessão ainda `UNKNOWN`; não atribuir a OOM/kernels, nem repetir Popen detached como se confiável. Método anterior usava processo Windows persistente chamando o launcher PowerShell.

Impacto confirmado: serviço8081 e experimento34731 sem listeners; download experimental interrompido; caches em RAM perdidos. Pós-checagem de27 arquivos de modelos/launchers não encontrou mudanças nos tamanhos/mtime/inodes disponíveis e hashes dos arquivos pequenos. Não é hash integral de todos os pesos nem auditoria de todos os dados do host. SSH ao Windows continua acessível. `lsof` no Mac não encontrou listener8002 nesta checagem; não havia snapshot local anterior suficiente para atribuir a ausência ao incidente. Nenhuma configuração SSH/túnel/firewall foi editada.

Ações após detecção: diagnóstico somente leitura; preservados manifestos, logs, arquivos parciais e evidência em evidence/incident-006.json. Nenhuma recuperação/relaunch/limpeza automática. Builder de benchmark já encerrou (reporta18 testes offline); não houve benchmark novo nem inferência abliterated. Revisão final desse reparo permanece pendente.

EXP-005 finalizado `INTERROMPIDO_POR_INCIDENTE`, H007 `UNRESOLVED`; não retomar/apagar o parcial sem nova autoridade. OPS-006 finalizado `INCIDENT`, não promovido.

Próxima autoridade necessária: aprovação explícita pós-incidente para recuperar o servidor8081 pela cadeia Windows preservada `start-shared-131k.ps1` → `start-shared-lmhead4-131k.ps1` → `start-server.sh`, mantendo um controlador Windows persistente, e retomar preparação/experimentos somente depois de observar estabilidade. Não alterar segurança nem registro/autostart do Windows como workaround.

## A06 — autorização explícita pós-incidente — 2026-09-08T21:09:12.623677+00:00
Usuário ativou goal `6da5c9f9-f5bb-4e19-98f5-71657549f7dc` autorizando expressamente recuperar API pelo launcher Windows preservado, corrigir persistência, verificar estabilidade e retomar experimentos não destrutivos. Isso remove o bloqueio anterior para esse incidente, não autoriza alterações de segurança/SSH, exclusões ou recuperação automática de outro incidente.

### REC-007 — recuperação via controlador Windows persistente
Pré-registro: revalidar ausência de servidor e integridade da cadeia original; usar criação de processo Windows independente da sessão SSH para chamar o launcher PowerShell preservado. H008: o controlador Windows mantém WSL/modelo vivo após encerrar SSH; alternativa: processo segue vinculado à sessão ou falha no startup. Refutação: controlador/API desaparecem depois de desconectar ou erro no log. Consultar documentação oficial do método antes da criação.
Escopo: somente nova instância da cadeia `start-shared-131k.ps1` na8081, sem mudar flags/pesos nem serviço SSH/firewall/registro/autostart. Log de recuperação novo dentro do diretório de tarefa Windows, sem overwrite. Confirmar PID/comando do controlador, API metadata e estabilidade em sessões SSH separadas, sem benchmark de carga no servidor compartilhado. Limite de startup10min, sem retry cego nem loop de relaunch.

#### REC-007A — resultado 2026-09-08T21:14:43.632104+00:00
Win32_Process.Create retornou0/PID38072, mas controlador não persistiu; log contém apenas duas linhas do shell antes do Python. Nenhum readiness/modelo carregado foi observado. H008 ainda `UNRESOLVED`. Falha limitada à tentativa de recuperação já autorizada; serviço já estava parado, nenhuma nova perda detectada. Não repetir igual. Fonte do primeiro comando preservada em evidence/recover-007-attempt-a.ps1.

#### REC-007B — hipótese discriminante de redirecionamento
A cadeia PowerShell usa ErrorActionPreference=Stop. Hipótese: capturar stderr nativo com `*>` converte avisos Python em erro PowerShell antes de gravá-los. Alternativa: processo provider/job não persiste independentemente da captura. Usar o mesmo Win32_Process.Create/launcher/cache scope, mas redirecionar stdout/stderr no cmd.exe (sem pipeline de objetos PowerShell). Predição: warnings passam a aparecer no stderr e servidor progride; se ainda morrer, logs diferenciados devem mostrar causa ou hipótese enfraquece. Diretório/logs novos, sem exclusões/retries automáticos; limite startup10min.

#### REC-007B — resultado
`OBSERVED`: stderr agora captura traceback: `zmq.error.ZMQError: ipc path ... is longer than 107 characters (sizeof(sockaddr_un.sun_path))`. Processo encerrou antes de servir; sem mutações em pesos. Hipótese de logging útil `SUPPORTED` para observabilidade, não prova que foi única causa de A. Causa atual exata: TMPDIR muito longo em /mnt/c.

#### REC-007C — correção de socket path
Pré-registro: manter o mesmo método/launcher, trocar somente raiz de cache/temp para novo `/tmp/qwen-r007-<UTC>` Linux com subdirtmp. Socket resultante cabe no limite107 e reside em Linux, não drvfs. Logs continuam novos no diretório Windows da tarefa. Nenhum diretório preexistente limpo/modificado. Previsão: ultrapassar init_ipc_channels e carregar pesos; se aparecer outro erro, registrá-lo antes de qualquer nova hipótese.

#### REC-007C — observação intermediária 2026-09-08T21:25:01.371188+00:00
Controlador Windows cmd.exe PID40628 continua vivo após várias sessões SSH desconectadas; parent6768. O startup ultrapassou IPC e carregou pesos/pools. Espera atual é compilação JIT de kernels FlashInfer em cache novo: `cicc`/`nvcc` ativos, três compiladores a100% CPU, 18GiB RAM disponíveis e sem swap significativo. Não é inferência lenta nem evidência de deadlock. Antes de atingir orçamento10min, ampliar limite desta recuperação para20min por evidência de progresso de compilação e isolamento do cache frio; não reiniciar/limpar caches.

Validação local do cliente de medição: revisão do orquestrador conferiu fixtures, ausência de sockets reais e tempfile run-owned; `PYTHONDONTWRITEBYTECODE=1 python3 -m unittest -v test_benchmark_stream` passou18 testes. Corrigidos finish_reason objeto, stats especulativas opcionais e último evento sem tokens novos. Nenhum benchmark ao vivo executado pelo cliente ainda.

#### REC-007C — resultado 2026-09-08T21:32:20.405612+00:00
`OBSERVED`: servidor pronto desde18:28:22 (-03); API via Windows localhost8081 retornou `/v1/models` com `qwen38`/131072 às18:30:03, após múltiplas sessões SSH desconectadas. Controlador cmd.exe40628, parent6768, vivo desde18:16:18. Snapshot atual em evidence/recovered-007.json. Método Win32_Process.Create + cmd redirecionando streams + TMPDIR curto foi executado com sucesso; kernel compilation em cache novo explicou ~12min de startup. Sem inferência benchmark no serviço compartilhado.

INCIDENT-006: indisponibilidade recuperada e persistência entre sessões observada. Correção operacional ainda precisa convergir para a inicialização final autoritativa ao promover o melhor perfil; não foi criado autostart/serviço/registro nem alterado SSH. O launcher preservado continua sendo a cadeia de origem. Causas específicas comprovadas nas tentativas de recuperação: logging nativo necessário para revelar erro e caminhoIPC>107bytes. A causa completa do descarte dos processos WSL no primeiro incidente permanece inferida, não confundir com a falha de socket subsequente.

Antes → Agora: Popen desacoplado somente dentro do WSL não manteve o experimento; controlador Windows WMI/foregroundWSL observado persistente, caches novos e IPC curto. Benchmark client revisado e18 testes offline passam.

### EXP-008 — baseline privado DFlash8 — 2026-09-08T21:42:01.259650+00:00
Pré-registro. Alvo Qwen NORMAL LMHead4 e draft DFlash2-NVFP4-RTNcal atuais, mesmos pesos e SGLang01e19ddf5. H009: perfil saudável produz contadores nativos completos, estabilidade em130048 tokens de entrada e recuperação exata de3 chaves sintéticas. Não assumir ≥200tok/s ou qualidade por intuição. Alternativa: capacidade/qualidade/métricas falham mesmo com janela nominal131072.

Configuração: DFLASH8/window2048, FlashInfer CUTLASS, KVFP8, GDNbf16/5slots, prefill4096, memfraction.92, max131072,1req. Diferenças instrumentais declaradas: porta privada34731, nome `qwen-normal-exp008`, seed7319, load-format safetensors, enable-metrics, paths e caches exclusivos. Nenhuma alteração de modelo/precisão/sampling para fabricar ganho.

Recursos: novo `/home/operator/qwen38-5090/optimization-runs/exp008-dflash8`; TMPDIR curto `/tmp/qwen-exp008/tmp`. Copiar apenas caches de compilação do REC-007C para arquivos independentes dentro do novo root; não compartilhar cache KV/processo/diretórios graváveis. Pesos e venv existentes somente leitura, offline HF, sem bytecode/credenciais/hooks/tools externos/HiCache. IO de downloads e outras builds não estará ativo durante medição; downloader NInfer segue parado.

Lifecycle: novo controlador Windows cmd/WMI executa `exec-manifest.py` foreground no WSL, aguardando PID/starttime do servidor recuperado384. Só então enviaremos SIGTERM ao servidor original revalidado. Isso mantém controladorWindows durante transição; não repetir PopenLinuxdesacoplado. Launcher experimental não é substituto permanente do launchercompartilhado. Novos scripts/manifests exatos preservados nos anexos.

Dados: scripts/make-cases.py cria 3tarefas (códigoLRU, matemáticaEuclides, prosaespeculação) × entradas2048/32768/130048tokens;3repetições porcaso, maxoutput1024, temp0/topk1/topp1, thinkingligado pelo template original. Cabeçalho sintético único porcaso minimiza prefixcache entrecasos; primeira repetição planejada fria e seguintes quentes, usando cached_tokens real para classificar. Qualidade separada: recuperar ALPHA=R17M9, BETA=V82K4, GAMMA=N53T7 em10/50/90% do contexto, uma requisição portamanho; output1024. Saídas de desempenho podem bater lengthlimit e NÃO provam correção de tarefa.

Métricas: TTFT, (tokens finais−primeirobloco)/(tempo último incremento−primeirobloco), E2E, aceitação e histogramas do servidor, VRAM/temperatura. Relatar porcaso e repetição; não misturar mediana geral como ganho. Fazer warmup separado e não contar compilation/startup como decode. Qualidade exige JSON final após thinking com3valores corretos, sem truncamento/erro.

Limites: startup reutilizando caches até10min, requesttimeout180s, suiteaté15min. Erro/protocolo/OOM interrompe suite; arquivos/estado antigos não são limpos; efeitos fora do run congelam operação. Sem promoção até comparações e validações adicionais de raciocínio/ferramentas.

#### EXP-008 — desvio de preparação 2026-09-08T21:57:09.825332+00:00
O startup ultrapassou o limite operacional previsto10min (observado657s). Isso é um desvio registrado, não sucesso do plano de reutilização de cache. Não houve benchmark. `OBSERVED`: nova compilação ptxas/nvcc ativa; seletorDFLASH greedy+sampling foi incorporado ao graph às18:53:17; captura progrediu e autotune adicional e7f26... começou18:53:51. Copiar caches para outro caminho NÃO eliminou recompilação. Hreuso-de-cache-por-cópia `WEAKENED`.

Decisão explícita de execução: ampliar esta preparação para20min pela evidência de progresso e recursos saudáveis, sem reiniciar ou limpar nada. Registrar startup real separado da medição de decode. Para próximas variantes, evitar relocação repetida de compiler cache; usar raiz estável de compilação pertencente à campanha, com KV/processo/arquivos de resultado separados e sem escritores concorrentes. Essa raiz não contém dados de usuário nem serve como rollback dos pesos.

#### EXP-008 — resultado 2026-09-08T22:11:32.405846+00:00
`OBSERVED`: qualityneedles passou3/3 (2048,32768,130048 entradas), todos os3valores corretos e stop normal. Medianas decode3reps: código2k271,39; matemática2k310,34; prosa2k153,75; código32k240,28; matemática32k207,23; prosa32k144,17tok/s. Isso NÃO é melhoria comparativa, é baseline do perfil atual com prompts controlados. Cada caso teve2saídas distintas entre3reps mesmo emtemp0; cache/numérica influenciam resultado, não declarar equivalência exata por seed.

Falha no primeiro desempenho130048: gerou1012tokens e abortou. Nativefinish_reason.type=abort, status500, mensagem `Out of memory even after retracting all other requests in the decode batch. Aborting the last request.` Cliente inicialmente mostrou erro secundário de usageausente; corrigido para preservar/exibir causa nativa antes de validar métricas.19testesoffline passam. Nenhum throughput foi inventado para esse casoabortado. Suite parou;9repslongas não completas. Servidor privado segue ready; GPUlivre2288MiB, sem reinício/crash. Falha ficou confinada ao request/cache experimental; não é nova perda de serviço externo.

Causa fundamentada: schedule_batch.py check_decode_mem verifica capacidade do KVallocator, e _new_tokens_required_next_decode_spec_v2 soma get_alloc_reserve_per_decode ao kv_committed_len antes da próxima verificação. Pool=janela131072 não deixa folga para olhar à frente no fim da sequência. `130048+1012=131060` abortou perto do teto, com VRAM ainda livre. Evidência em evidence/analyze-008.txt e evidence/kv-boundary-source.txt, resultados brutos no diretório exp008/results. H009: qualidadepontual `SUPPORTED`, capacidade de preencher orçamentointeiro `REFUTED` neste perfil.

Aceitação explica a diferença por tarefa: ~4,28tokens/ciclo emcódigo2k,4,88emmatemática,2,41emprosa; divididos pelas velocidades observadas, ciclos~15,7–15,8ms. Isso aponta a draft/aceitação antes de atribuir a prosa lenta a outro kernel. Não extrapolar a todos os workloads.

### EXP-009 — folga física de KV sem reduzir contexto
Pré-registro. H010: adicionar256tokens de folga ao pool (`max_total_tokens131072→131328`) mantendo `context_length131072` e todas as flags/modelos constantes elimina o abortono limite. Refutação: mesmo request130048+1024 aborta ou janela/qualidade regride. CustoKVtarget estimado8MiB, muito menor que2,2GiBlivres; confirmar apósboot. Não é redução de contexto nem relaxamento de verificação.
- Novo processo/resultado/cwd exp009-kv-reserve, porta34731 e nomeqwen-normal-exp009; mesma suite, seed7319. RootTMPcurto novo.
- Compiler caches permanecerão na raiz da campanha criada em exp008 (apenas códigos/autotuning, semKV/usuários); sem escritores concorrentes, servidor008 encerra antes de009 iniciar. Isso evita a recompilação porrelocação observada; compartilhamento é só entre experimentos da mesma campanha, não caches preexistentes do usuário.
- Executar primeiro a mesma suite de retenção, depois3reps/caso deperformance. Exigir1024tokens/stopválido nos casoslongos, registrar cached_tokens e hashesdesaída. Benchmarktotal15min; startup20min máximo, com cacheestável espera-se <2min. Qualquer falha encerra suite e será registrada.

#### EXP-009 — resultado 2026-09-08T22:28:09.356219+00:00
`OBSERVED`: startup94,30s usando compiler cache em caminhoestável (antes708,51s comcópiarelocada). Qualityneedles3/3;27requisições deperformance terminaram semabort. H010 `SUPPORTED`: folgaKV eliminou OOM de pool. Porém todos oscasos130048 retornaram1022tokens, não1024; finish_reason.length=1022 mostra clamp delimitedosaída. Critério deorçamentointeiro ainda NÃO atendido. Medianaslongas: código213,59; matemática204,91; prosa122,94tok/s; dados não equivalem a benchmarkdequalidade das saídas truncadas. Suite301,58s, artefatos exp009/results e evidence/baseline-009-summary.txt.

### EXP-010 — janela utilizável com margem de enquadramento
Pré-registro: mudar somente context_length131072→131200, mantendo pool131328 e orestantedo009, para acomodar guardas internas sem reduzir o orçamentodousuário131072. H011: entrada130048+saída1024 passa semclamp/abort; alternativa: outro limiter ainda restringe. Pool tem128tokens adicionais à janela131200; reservaespeculativa permanece. Modelo nativo262144, semextrapolação/RoPEcustom. Inputs/reps/sampling/client idênticos ao009.
Novoexp010-context, modelo servidoqwen-normal-exp010, TMPDIRnovo curto, compiler cachecampanhaestável. LifecycleWindowsjávalidado; repetir mesma suite e conferir explicitamente token counts. Mesmos limites de15min/suite e20min/startup (esperado<2min). Nenhum ganho de velocidade atribuído à simplesmudançadolimite.

#### EXP-010 — resultado 2026-09-08T22:45:14.345793+00:00
`OBSERVED`:27/27 requisições deperformance retornaram1024tokens, inclusive9/9 comentrada130048; nenhumafoiabortada/clampada. Qualityneedles3/3 novamente. H011 `SUPPORTED`: context131200/pool131328 suporta orçamento utilizável131072 mantendo margem interna. Startup99,58s; suite300,70s. Medianasdecode: código2k270,14/matemática2k309,35/prosa2k155,10; código32k237,54/matemática32k207,56/prosa32k144,47; código130k215,17/matemática130k204,07/prosa130k122,81tok/s. Não é speedup comprovado sobre008; é correçãodecapacidade/estabilidade.

### EXP-011 — raciocínio e ciclo de ferramenta
Pré-registro: chamar somente a APIprivada exp010 na34731, mesmoPID/modelo/caches próprios, sem redeexterna/proxies/redirects. Quatro oráculosdeterminísticos: produto121401, gcd21, probabilidade5/14, CRT17; thinking habilitado e max_tokens4096. Exigir JSONcorreto, stopnormal e observar campo de raciocínio. Ferramenta sintética add_numbers(17,25), validar chamada/args, fornecer resultado42 sem executar ferramentaexterna, exigir resposta final{sum:42}. H012: parsers+raciocínio+ciclo2turnos permanecemfuncionais. Falha para e registra, sem mudarprompt para esconder. Limite180s/request,7requestsno máximo. Saídas/logs exclusivamente emexp010/results/quality-api.jsonl; semcódigo gerado executado. Isso é avaliaçãofocada, não provaqualidadegeral.

#### EXP-011 — resultado 2026-09-08T22:58:56.553948+00:00
Os quatro oráculos matemáticos passaram, com rastros de raciocínio de476/308/512/377 caracteres. Chamada estruturada add_numbers(17,25) validada; resultado sintético42 devolvido ao modelo; resposta final sum=42 passou. Evidência: evidence/quality-api-011.txt e exp010/results/quality-api.jsonl. H012 SUPPORTED no escopo focado; não é avaliação ampla de programação ou conhecimento.

### OPS-013 — retomar download NInfer com pausa cooperativa
Pré-registro de aquisição: continuar o arquivo parcial de3.753.902.080 bytes, que foi criado pela tarefa e não foi carregado. Fonte/revisão/hash iguais ao EXP-005; não retomar abliterated. Novo controlador Windows WMI mantém python foreground no WSL, independentemente da sessão SSH.
Script resume-ninfer-download.py verifica arquivo regular sem symlinks/hardlinks, tamanho inicial esperado, lock exclusivo e resposta HTTP206/Content-Range exatos por intervalo64MiB. Apenas acrescenta bytes; não trunca/apaga nada. SHA256 final deve ser552c374c685dce302603b95fbe940fb04243c0cd44c083efc644ad3d980d462c, tamanho23.719.496.192.192? Correção imediata: tamanho correto é **23.719.496.192 bytes**, conforme manifesto; o número anterior com sufixo duplicado não é usado pelo script.
Root de controle exclusivo: /home/operator/qwen38-5090/optimization-runs/ninfer-resume-013. control.json permite pausa entre intervalos sem manter socket aberto. Não medir performance até status PAUSED/COMPLETE. Ambiente sem credenciais/proxies; rede somente para o artefato público fixado. Limite3h por tentativa, timeout120s/leitura, sem retry automático em erro/hash inválido. Evidência e parciais preservados se falhar.

### EXP-012 — DFlash bloco6 versus8
Pré-registro. Fonte instalada dflash_worker_v2.py linhas329–348 usa speculative_num_draft_tokens como block_size e chama draft_model.set_block_size. Alterar os dois flags correlatos para6 é uma variável: comprimento do bloco. Não alterar pesos, sampling, contexto/pool ou thresholds de aceitação.
H014: bloco menor reduz custo de draft/verify e melhora prosa, mas pode diminuir aceitação em código/matemática. Rejeitar como default geral se ganho geométrico das medianas porcaso <5%, regressão >10% em algum caso ou qualquer falha de qualidade/contexto. Mesma suite27 medições+3needles, baselineEXP010. Separar cache frio/quente pela telemetria; não tratar repetições como i.i.d. Eventual vencedor exige confirmação; pico isolado não vale promoção.
Root novo exp012-block6/TMP curto, porta34731/modelo qwen-normal-exp012. Compiler cache estável, um servidor porvez. Download NInfer explicitamente PAUSED/COMPLETE antes de medir. Startup20min máximo; suite15min. Aviso de tamanho treinado8 é esperado; erro real encerra experimento.
Preparação inicial falhou localmente por string Python não fechada, antes de escrever/usar qualquer arquivo remoto. Corrigida a sintaxe, sem alteração de hipótese ou recurso.
Registro UTC: 2026-09-08T23:11:14.516274+00:00

#### EXP-012 — resultado 2026-09-08T23:21:55.564872+00:00
Todos os casos terminaram, incluindo orçamento131072 e3/3needles. Bloco6 versus8: código2k−5,59%, matemática2k−13,80%, prosa2k−0,43%; código32k−3,92%, matemática32k−0,77%, prosa32k+3,26%; código130k−9,15%, matemática130k−15,75%, prosa130k+5,81%. Média geométrica−4,74%. H014 como melhoria geral REFUTED; rejeitado para default. Ganho localizado em prosa longa não justifica regressões gerais. Melhor referência permanece EXP010/bloco8; o processo012 é somente experimental. Suite310,83s.

OPS-013: download confirmou PAUSED em14.357.102.592bytes antes da suite012; agora pode retomar fora de medições.

### OPS-014 — toolchain NInfer isolada — 2026-09-08T23:33:02.277901+00:00
Fonte NInfer fixada b88c0f6fc7e999f13eb2fcf7fc9105ed79a91868, arquivo10.218.520bytes SHA256866f3acbc21f2a9c2cc90fbed49ec4e7cd14b5890126ac0ac3bcca31c2af11f1, inspecionado antes da build. Root/src/apps CMake lidos: appsON, testes/benchmarksOFF, bibliotecas estáticas e spdlog vendorizado; restante dos hooks em verificação read-only pelo scout. Nenhuma compilação iniciada antes desse fechamento.
Instalar somente ferramentas em novo /home/operator/qwen38-5090/optimization-runs/ninfer-build-014: venv CUDA13.1 (nvcc/crt/nvvm/nvtx/cccl13.1.115, runtime13.1.80), CMake3.31.10 e Ninja1.13.2. Usar uv existente com cache próprio, sem configs/credenciais herdadas, somente wheels e versões fixadas. Não alterar venvSGLang/CUDA13.0/driver.
Dependências Ubuntu serão APENAS baixadas com APT em estado/cache/log/status-dpkg copiados para a raiz própria e extraídas com dpkg-deb -x em sysroot próprio; nunca apt install real, sudo, dpkg -i, hooks globais ou atualização do sistema. Fontes confirmadas archive.ubuntu.com e security.ubuntu.com. Verificar hash do statusdpkg global antes/depois. Precisamos pkgconf e headers/runtimes FFmpeg/libcurl conforme Dockerfile oficial; nomes/pins/resolução nos logs.
Limites: instalação30min, downloads≤2GiB nesta etapa, filesystem próprio, nenhuma inferência/modelo carregado; não medir GPU enquanto houver build/download ativo. Scripts exatos preservados. Build futura somente targets ninfer/ninfer-serve, Release/sm120a, dois jobs, até60min, após validar pré-requisitos e hooks.

#### OPS-013/014 — aquisição e setup concluídos — 2026-09-08T23:42:25.096542+00:00
Download NInfer COMPLETE:23.719.496.192bytes, SHA256552c374c685dce302603b95fbe940fb04243c0cd44c083efc644ad3d980d462c. Toolchain instalada em raiz própria: nvcc13.1.115/CMake3.31.10/Ninja1.13.2.64pacotes Ubuntu somente baixados/extraídos; hash do statusdpkg global permaneceu idêntico. Smoke de versões passou: libavformat60.16.100, avcodec60.31.102, avutil58.29.100, swscale7.5.100, curl8.5.0. SGLang e driver não foram atualizados.

Scout fechou os CMake restantes: targets qwen3_6/qwen3_6_27b/qwen3_6_35b_a3b só adicionam fontes/includes; spdlog só biblioteca estática/Threads; nenhum custom command/execute_process/download/install/test automático e nenhum arquivo.cmake no escopo. Build explícita ninfer/ninfer-serve com testes/benchmarksOFF não executa modelos.

### BUILD-014 — compilar NInfer sem alterar ambiente existente
Pré-registro: fonte e dependências verificadas, root exclusivo ninfer-build-014; build novo, Release/sm120a, dois jobs, targets ninfer/ninfer-serve. Variáveis de cache/home/operator/log apenas próprias; CUDA13.1 pelo caminho explícito, bibliotecas FFmpeg/curl no sysroot próprio, libcuda do WSL é entrada somente leitura. Criar apenas symlink libcudart.so dentro do SDK novo se necessário para CMake. Não tocar no SDK13.0 existente. Limite60min para compilação, logs/status preservados, sem execução de benchmark/app nem GPU de inferência. Qualquer erro será diagnosticado antes de novo comando; sem remoção/limpeza.

#### BUILD-014A — falha de layout SDK — 2026-09-08T23:55:15.647145+00:00
CMake falhou em0,72s no teste de identificação do compilador, antes de compilar NInfer. nvcc.profile procura lib64, mas as wheels guardam libcudadevrt.a/libcudart_static.a em lib. Erros: cannot find -lcudadevrt e -lcudart_static. Os arquivos existem no SDK próprio, conforme smoke. Sem GPU/modelo executado, sem mudança global.

BUILD-014B pré-registro: acrescentar LIBRARY_PATH apontando somente ao lib do SDK próprio e /usr/lib/wsl/lib (entrada somente leitura), preservando fonte/versões/flags. Hipótese: g++ encontra as bibliotecas estáticas mesmo com defaults lib64 do nvcc. Reconfigurar a árvore de build própria sem apagar nada; manifest/log anterior preservados. Mesmos limites de60min e2jobs, nenhuma limpeza/instalação global.

#### BUILD-014B — resultado e nova evidência — 2026-09-09T00:00:57.995135+00:00
LIBRARY_PATH resolveu a detecção CUDA13.1 e encontrou FFmpeg/curl. Depois CMake falhou com "no known features for CXX compiler GNU13.3.0". Inspeção de CMakeCXXCompiler.cmake e CMakeCCompiler.cmake mostra COMPILE_FEATURES vazio, COMPILER_WORKS/ABI_COMPILED vazios e COMPILER_ID_RUN=1: cache parcial deixado pela primeira falha interrompeu detecção. Não é evidência de GCC13 sem suporte C++20.
BUILD-014C pré-registro: manter LIBRARY_PATH corrigido e reconfigurar em build-c NOVO, preservando build antigo e logs; não apagar cache nem forçar capacidades do compilador. Predição: completar detecção C/CXX/CUDA e iniciar compilação. Mesmos source/pins/limites; sem alteração de kernels ou sistema.

### BUILD-014C resultado / cliente comum
NInfer compilado em662,60s; ninfer-serve SHA256 fbf2b0e645e1dd96fc25d173d425ab65939febac33cb664cd0c395f7bd215d8a. --help executou sem inicializar servidor/modelo (main.cpp retorna antes). Cliente revisado: protocolo nativo preservado; OpenAI usa usage final, TTFT e estimativa explicitamente enviesada, decode exato=null.25 testes offline repetidos com sucesso; apenas FakeOpener e temporários próprios.
SMOKE-015 pré-registro: verificar cliente OpenAI em processo exp012 de identidade PID/start/boot própria, só texto sintético128tokens; resultados em client-smoke-015 novo. Nenhum recurso externo/media/tool; modelo/KV já pertencem ao experimento. Não medir ganho com esse smoke. Modelos e configurações existentes somente leitura, nenhum reset/cleanup.

SMOKE-015 falhou no cliente: reasoning_content=null no primeiro delta de papel do SGLang. Captura SSE real confirmou. Correção limitada: null/missing tratados como sem texto; valores não-string continuam erro; regressão offline adicionada,26 testes PASS. Repetir em client-smoke-015b novo; preservar falha original.

### EXP-016 — referência bloco8 pelo protocolo comum
SMOKE-015B passou ao vivo: conteúdo42 com reasoning separado, usage42; estimativa não é decode exato. Erro intermediário de geração do script (parêntese) ocorreu antes de executar qualquer Python remoto e foi corrigido/verificado com ast.parse.
Pré-registro016: mesmo perfil validado010 (bloco8/context131200/pool131328), nova identidade/estado exp016-common8, mesma cache de compilador do projeto sequencialmente. Comparação comum: mensagens semanticamente iguais extraídas dos IDs originais; cada backend aplica seu template, portanto registrar tokens reais de entrada, não afirmar igualdade exata.3tarefas x3contextos x3repetições,1024saída greedy thinking habilitado;3needles adicionais. Medidas: TTFT/E2E e stream estimate explicitamente enviesada; usar native anterior separadamente. CPU build/download completos antes de medir.
Transição016: controlador Windows aguardando identidade PID/start/boot5009 da variante6; SIGTERM somente esse processo experimental após confirmar controlador aguardando. Sem reboot/alteração de launcher compartilhado/modelos/SSH/túnel. Port34731 privado, filesystem/tmp/KV sintético próprios; sem remoção/cleanup.

### EXP-015 — NInfer DFlash2/8
Reconciliação de recursos: scout confirma modelo O_RDONLY/MAP_PRIVATE/pread, caches Responses/mídia em RAM, logs operacionais stderr e única gravação persistente da aplicação=request-log explícito. Nenhum fetch automático no caminho somente texto; mídia desativada e payloads sem URLs/imagens. CUDA cache apontada a root próprio e desabilitada. Header do arquivo verificado sem carregar GPU: identity qwen3.8-27b/nvfp4;66 objetos dflash2 e12mtp, além do SHA256 previamente completo. Assim os dois gaps do scout foram resolvidos para lançar esse artefato.
Contrato: root exp015-ninfer8 e /tmp/qwen-exp015 novos, loopback34732, concurrency1, contexto131200/KV131328 FP8, prefill4096, dflash2 draft8, greedy, pensamento habilitado por padrão (sem --no-thinking), preserve-thinking. Pesos/SDK originais somente leitura; nenhum serviço externo/credenciais/context preset; somente RAM/GPU e filesystem próprios descartáveis, sem cleanup. Executor existente agora permite também SOMENTE o binário NInfer de SHA fixado, preservando caminho SGLang.
Comparação é runtime+export+configuração especulativa, não runtime puro. Referência016 comum completou27/27 saídas1024 com130086tokens de entrada longa (131110total). usage.reasoning_tokens do SGLang às vezes supera completion_tokens; não usar esse subcontador como oráculo. Contagem completion_tokens e métricas brutas preservadas.

EXP015 falha controlada de inicialização: kv_capacity fora do intervalo, antes de carregar pesos; processo encerrou e GPU voltou a30440MiB livres. Evidência layouts_impl.h:646-673: com concurrency1, páginas KV precisam ser iguais às páginas do contexto. A reserva adicional necessária no SGLang não é transferível ao NInfer.
EXP015B pré-registro: somente reduzir KV131328→131200, mesmo contexto131200, DFlash2/8 e demais flags. Novo root/tmp/identidade/logs; preservar015 como evidência. Ainda ≥131072 utilizável será testado, não assumido.

EXP015B passou inicialização13s com21,1GiB de pesos eKV131200FP8. Suite comum completou27/27 respostas1024 inclusive130086entrada=131110total, sem corte/abort. Ainda não promover: desempenho e qualidade em consolidação. Próximo teste usa mesmos4oráculos matemáticos+tool add_numbers sintético/retorno42 do011, via Chat padrão, max4096; sem chat_template_kwargs específico, pensamento padrão do NInfer habilitado. Ferramenta é somente schema e mensagem de resultado emulada, nenhuma função/model-code executada. Recursos continuam processo/KV/logs próprios; nenhum serviço externo.

EXP015B consolidado: needles3/3 comJSON exato (ALPHA R17M9/BETA V82K4/GAMMA N53T7);4oráculos matemáticos e emissão/retorno de ferramenta PASS, raciocínio presente. Ambas engines tiveram as MESMAS contagens de prompt no protocolo comum. NInfer8 vs SGLang8: média geométrica de razões das medianas de stream-estimate -8.73%; não supera referência geral. Cache-hit TTFT NInfer~36–66ms, menor que SGLang~110–291ms, mas geração longa math/prosa pior. Backend request_done expõe decode/GPU-host timing e aceitação, útil para próxima hipótese.

EXP017 pré-registro: NInfer DFlash2 draft4 contra draft8, única alteração comportamental; menos verificação por rodada pode compensar propostas profundas rejeitadas. Mesmo artefato/binário/contexto/pool/prefill/cache e suite. Novo root/tmp/identidade; guardar todas falhas. Próximos candidatos: MTP3, depois profiling nos gargalos medidos; nenhum kernel será alterado sem dados.

EXP017 resultado: suite completa sem erro; draft4 ficou−3,89% na média geométrica das medianas contra NInfer8. Ganhos locais em prosa130k(+10,69%) e math130k(+5,09%), mas perdas code32k(−17,13%) e code130k(−15,38%). Rejeitado como padrão geral.
EXP018 pré-registro: comparar backend MTP draft3, bundle12objetos confirmado no mesmo arquivo. Demais parâmetros/casos/recursos preservados. Novo root exp018-mtp3/tmp/identidade/logs, controlador aguardando somente processo017 verificado porPID/start/boot antes deSIGTERM. Nenhuma mudança global, exclusão ou código de kernel.

EXP018 resultado: MTP3 completou a suite, porém −30,92% contra SGLang8 (média geométrica das razões de medianas stream-estimate). Não é candidato a padrão de velocidade; economiza VRAM (~27,3GiB usados vs29,2GiB DFlash2). Não repetir qualidade extra em candidato já rejeitado por desempenho.

### EXP019 — profiling da referência SGLang8
Voltar ao perfil016 inalterado em identidade/root019 novos; encerrar só o processo MTP018 após controlador persistente aguardar PID/start/boot verificados. API continua privada34731; nenhuma alteração SSH/túnel/produção. Runtime/SDK/cache compilação anteriores preservados. Pré-registro: observar kernels/intervalos CPU-GPU da referência antes de otimizações CUDA/Triton; não tratar tempo bloqueado esperando GPU como prova de saturação. Endpoints /start_profile e /stop_profile existem, DTO ProfileReq permite output_dir/num_steps/activities CPU,GPU/with_stack/record_shapes. Artefatos deverão ficar somente em exp019-profile8/profiles; nunca profiling MEM/snapshot de pesos. Primeiro fechar caminho de exportação/profiler e obter traces reais, depois escolher hotspot.

PROFILE019 closure: profiler_manager usa output_dir explícito, exporta .trace.json.gz em stop; num_steps32 conta a partir do forward atual(+33), auto-stop exporta. PROFILE_V2 está UNSET no manifest e defaultFalse em environ.py:435, portanto caminho auditado ativo. CPU/GPU apenas, merge=false, semMEM/RPD/stack; defaults/tmp externos não usados.
Coleta: warmup64tokens no mesmo prompt nativo, depois capturar32steps em geração1024 de code2k e prose130k (prefix aquecido para evitar trace ocupado só porprefill). Saídas/requests/traces exclusivamente root019; medições com profiler NÃO serão usadas como throughput competitivo. Hipótese: custos de kernels ou lacunas entre launches explicam limite; nenhuma alteração de kernel antes de decompor trace.

PROFILE019 coletado: dois traces reais CPU/GPU exportados,~174k/176keventos. Cada trace inclui1EXTEND (2k ou3072tokens) e31rodadas especulativas. Portanto excluirprefill da análise de decode; não confundir o kernel longo mask0 inicial com gargalo de geração. Análise porintervalos GPU step[TARGET_VERIFY] (62spans, draft+target) salva em evidence/profiles019/decode-summary.json.
exp019-code-2048-TP-0.trace.json.gz: 38537 kernels de decode, soma447.29ms dentro de802.46ms de spans GPU. Top kernel representa32.75% do tempo somado. Traces instrumentados não são benchmark de throughput.
exp019-prose-130048-TP-0.trace.json.gz: 38537 kernels de decode, soma554.36ms dentro de1005.59ms de spans GPU. Top kernel representa26.50% do tempo somado. Traces instrumentados não são benchmark de throughput.

Correção análise019: há62anotações CPU, mas93spans GPU, alguns sobrepostos. Não somar spans brutos. Recalcular por UNIÃO de intervalos para evitar dupla contagem/exclusão indevida. JSON anterior preservado como decode-summary-initial.json; decode-summary.json corrigido é o resumo vigente.
exp019-code-2048-TP-0.trace.json.gz: união62intervalos GPU, 39715kernels, soma477.74ms/união468.71ms; top3percentuais [31.49, 28.05, 5.72].
exp019-prose-130048-TP-0.trace.json.gz: união62intervalos GPU, 39715kernels, soma592.01ms/união565.26ms; top3percentuais [25.48, 22.74, 19.87].

### CHECKPOINT — 2026-09-09T01:14:13.740683+00:00
019 está funcionando como qwen-normal-exp019 na porta privada34731 após exportar perfis; API compartilhada8081 ainda em manutenção. NInfer DFlash2/8, DFlash2/4 e MTP3 medidos; nenhum supera o SGLang8 no conjunto. NInfer8:≥131072, needles e matemática/tools focados passaram.17/18: needles3/3 e27/27 gerações1024, total longo131110, mas sem necessidade de testes extras de qualidade por rejeição de velocidade.
Profiling real coletado e separado do prefill: os dois kernels NVFP4 CUTLASS principais somam~59,5% dos tempos de kernels no decode curto e~48,2% no longo; atenção paginada é~19,9% no longo. Somar durações de kernels pode exceder tempo de união dos spans por concorrência; NÃO interpretar soma como latência exclusiva/saturação. A análise inicial errada de spans está preservada e corrigida explicitamente.
Próxima decisão fundamentada: mapear seleção/tuning desses GEMMs no SGLang/FlashInfer e comparar backends/configurações CUDA suportadas mantendo pesos/qualidade. Atenção longa é segundo alvo. Ainda faltam otimizações orientadas pelos traces, validação do melhor perfil final e promoção compartilhada/estabilidade final; objetivo NÃO concluído.

### EXP020 — cuDNN FP4 orientado pelo profiling — 2026-09-09T01:25:18.750008+00:00
Fatos: SGLang aceita flashinfer_cudnn paraNVFP4; FlashInfer declaraSM120; cuDNNfrontend1.28/backend9.20 instalados satisfazem mínimos1.13/9.10.2. MesmoW4A4, quantização de ativações e layout/padding doCUTLASS; sem regravar checkpoint. Cublas direto não é opçãoSGLang; CuTe/TRTLLM expostos limitados aSM100/103/107.
Hook KDA resolvido: módulo é gemm/__init__.py, não gemm.py. Chama kda_kernels/qwen3x_nvfp4_gemm.py. Shapes de5120/17408 dos DOIS modelos não são elegíveis emM8; exceção elegível(M9,K17408,N5120) motiva candidato bloco9 depois. Não bloquear cuDNN com hipótese de interceptação geral. FlashInfer também tembackend b12xSM120/CUDA13 não exposto noSGLang; candidato de integração posterior, sem escolher implementação ainda.
Hipótese020: mudar somente --fp4-gemm-backend flashinfer_cutlass→flashinfer_cudnn reduz custo dos GEMMs dominantes, podendo melhorar ≥5% no conjunto. Deixar autotuning habilitado. Pesos, DFlash8,131200contexto/131328pool,FP8KV, memória e parsers preservados. Mesma suite comum016. Pré/pós qualidade e contexto reais obrigatórios antes de promover; duração instrumentada não é métrica competitiva.
Recursos: novo exp020-cudnn8 e/tmp/qwen-exp020, KV/RAM/GPU sintéticos próprios; cache de compilador da campanha reutilizada sequencialmente, HOME/XDG/armazenamento próprios; cuDNN plano/workspace emRAM, nenhum download/instalação/globalconfig. Somente localhost34731 e texto sintético; não usar produção8081/8002. Controlador Windows aguarda019 verificado antes deSIGTERM único. Limite inicialização900s se primeira seleção de planos demorar; sem limpeza/remoção.

EXP020 resultado: cuDNN inicializou e completou suite, mas −24,37% contraCUTLASS8 no conjunto comum. Todas9células mais lentas; rejeitado.
### EXP021 — cobertura do kernel KDA em bloco9 — 2026-09-09T01:39:24.260752+00:00
Hipótese: voltar CUTLASS e usarDFlash9 aciona kernelCuTeDSL SM120 já integrado para(M9,K17408,N5120), forma exata do down_projection dos modelos target/draft (config.json5120/17408,64/5layers). Não afirmar ativação apenas pornome: validartrace apósganho e qualidade. Mais propostas e kernel especializado podem superar bloco8; medir suiteigual e gates decontexto/qualidade. Não modificar checkpoint/quantização.
KDA dispatch gate e arquivo qwen3x_nvfp4_gemm_sm120.py inspecionados: compila cute.compile e mantém cache emRAM; sem open/download/subprocess aplicativo. CuTeDSL env_manager suportaCACHE_DIR/DUMP_DIR; definir CUTE_DSL_CACHE_DIR eCUTE_DSL_DUMP_DIR sobnovo root021, sem cache externo. CUDA_HOME permaneceSDKoriginal. Demaisrecursosiguais, tmp/root/identidade novos, controlador persistente aguardando somente020 verificado antesSIGTERM. Limiteprimeiracompilação900s; sem cleanup nem alteraçõesglobais.

EXP021 resultado inicial: conjunto comum−4,13% contra8; código32k+23,76%, mas maioria dascélulas pior. Não promover. Coletar trace32steps de code2k para comprovar seKDAfoi de fato ativado; não atribuir causalidade somenteaosflags.
### MICRO022 — hipótese localizada, antes de qualquer alteração de gate — 2026-09-09T01:57:25.780794+00:00
O kernelKDA subjacente compila M/K/N simbólicos, mas gate27B só aceitaM9. Testar diretamente M8 contraCUTLASS com operandosNVFP4 sintéticos idênticos, e variante cache_policy=True já existente, sem editar pacote/gate/kernel. Candidato de integração depende de melhoria L2-frio e correção. Script em construção local pelo builder; nenhuma execução remota delegada.
Oráculo pré-definido: finitos, erroL2normalizado≤1e-3 e |erro|≤0,015625*max(|referência|,1) por elemento. M8/M9,K17408,N5120,seeds17/29/43. Não é prova de equivalência bitwise nem qualidade de modelo. Graphs comflushL2de256MiB antes da região cronometrada eCUDAevents externos;50replays intercalados, sem timinghost como substituto. Erro encerra e preserva resultado; nenhuma flexibilização pós-falha. Execução só após revisar script e recursos.

MICRO022 readiness — 2026-09-09T02:15:24.318684+00:00
Revisão corrigiu captura para torch.cuda.graph (stream próprio/instanciação automática), igualdade bitwise real, preheat separado, rejeição de tempos inválidos e todos casos numéricos antes dos timings. AST/--help PASS. Erro de aspas no gerador local impediu primeira preparação antes de qualquer ação remota; gerador corrigido com AST.
Novo micro022 e /tmp/qwen-micro022; HOME/CUDA/CuTe/cache/dumps próprios, compiler cache da campanha reutilizada sequencialmente. Nenhum modelo será carregado, somente matrizes sintéticas/graphs/flush256MiB; >=27000MiB livres exigidos. Serviço21 pausado somente após controlador aguardando PID/start/boot correto. Executor port=null significa job semAPI. Sem rede externa/credenciais/reset/cleanup; bibliotecas e modelos protegidos somente leitura. Liberação somente da memória do próprio processo. Limite900s; nenhuma alteração de gate autorizada por resultado presumido.

MICRO022A — 2026-09-09T02:19:23.768638+00:00
Falha antes de comparar resultados: variantecache_true passava Btransposta ao kernel debaixo nível, que exige B[N,K/2]; FFI recusou com Mismatched mB.shape[1]. Gate não alterado; semtimings válidos. Corrigir somente bq.T→bq nessa chamada paraigualar_run; valores/pointers/oráculo preservados. Repetir em micro022/b comnovoresultado/logs; preservar022A/scriptoriginal. Processoanterior encerrado, semserviçoGPU concorrente;mesma exclusividade>=27000MiB/cache própria.

Preparação022B: WSL reiniciou após todos osjobs encerrarem; /tmp foi limpo pelo ciclo normal daVM. Subdiretório persistente022/b havia acabado de ser criado vazio, mas mkdir doTMP filho falhou porqueopai temporário desapareceu. Confirmar022/b vazio e criar árvoreTMPprópria comparents=True; nenhuma inferência/mutação de modelos executada nessa falha.

MICRO022B resultado confirmado em JSON bruto: PASS,9 comparações/6 casos M8/M9×3seeds, erro L2 e máximo zero. M8 CUTLASS mediana~0,06474ms; KDA~0,03402ms (−47,4%); cache_policy=True sem ganho adicional. Comparação sintética contra dispatch CUTLASS deste script, NÃO garante mesma tática autotunada do servidor e NÃO demonstra ganho na conversa. Fonte original do gate sha256=b14c3eaa6b8d9d62b559a50be62fe4cad935e7172bdb79d18a1f6abcb4ccb7c5.
EXP023 hipótese: adicionar somente (8,17408,5120) à whitelist do gate em processo isolado; manter todas verificações de dtype/layout/device e fallback, kernel inalterado/cache_policy padrão. Usar sitecustomize próprio via PYTHONPATH exclusivo do manifest023, com hash do módulo obrigatório e encerramento fail-closed se divergir. Não alterar pacote instalado/modelos/SSH/túnel; retirar PYTHONPATH reverte integralmente. Verificar ativação via trace e suite comum, depois qualidade/contexto antes de promoção. Recursos: root023 e/tmp/qwen-exp023 novos, CUDA/CuTe/HOME/storage próprios, cache compilador da campanha sequencial; API privada34731, só texto sintético e peso readonly.

EXP023 suite concluída: média geométrica das medianas versus016=1.062%. Todas27 gerações1024, incluindo130086+1024=131110 tokens. Microganho não se traduz automaticamente no servidor autotunado. Próximo: trace de ativação/custo e comparação direta da referência no mesmo ciclo para separar ruído/tática.

EXP023 qualidade focada: quatro oráculos matemáticos e tool roundtrip sintético passaram com raciocínio observado. Trace code2k confirma2139 launches KDA; kernel downprojection agora~68,75ms somados (inclui contagem do trace, não throughput). Ganho conjunto+1,06%, não47%; referência micro usa seleção diferente do servidor autotunado. Não promover nem alegar equivalência ampla: faltam controle contemporâneo, amostragem estocástica e demais oportunidades b12x/atenção/draft. Serviço023 ativo somente34731,8081 permanece manutenção.

EXP024 pré-registro: controle contemporâneo SGLang8 sem hook, mesma configuração016 e suite. Medir se+1,06%023 excede variação temporal; restaurar referência privada para próximo ciclo. Root024/tmp próprios, mesmos pesos read-only e cachecompiler sequencial;CUDA/CuTe explícitos. Pausar somente023 após controller024aguardarPID/start/boot corretos; não tocar8081/túnel. Scout separado somente-leitura investiga semântica de amostragem DFlash; nenhuma modificação estocástica nestecontrole.

UNSTUCK024 2026-09-09T02:55:55.632417+00:00 — OBSERVED: controle024 falhou no bind34731 antes deGPU/modelo, sem server-pid.json; último ss não mostrou listener34731. RETRIEVED: docsPython3.12 documentam Errno98 porTIME_WAIT. UNKNOWN: estado históricoexato. Nenhum reinício/correção executado após freeze. Notebook .pi/recovery/unstuck/20260909T025340Z-control024-bind.md BLOCKED aguardando autorização pós-falha para reprodução TCP efêmera, correção limitada do launcher e reinício privado. Coleta read-only das fontes de amostragem prossegue; objetivo global incompleto.

OBSERVED/INFERRED — diagnóstico024 do harness: reciboPID só nasce após bind; watcher não detecta falha anterior ao recibo e sai0 mesmo após timeout900s. Isso explica espera inútil e tentativa subsequente debenchmark semservidor. Correção ainda não executada: freeze pós-incidente. Read-only coleta de5fontes deamostragem concluída; scout921a1a49-00f8-4fc analisa helpers sem executar aplicação.

## Continuation7 — barreira persistente
- OBSERVED: continuações5,6,7 repetiram objetivo original sem resposta explícita à autorização pós-incidente024. Nenhuma correção/reproduçãoTCP/reinício autorizada após os fatos.
- INFERRED: execução do objetivo exige API experimental funcionando; repetir startup ou trocar método sem autorização violaria freeze. Encaminhar goal_blocked, repeated_turns=3. Não declarar otimização concluída nem oportunidades esgotadas.
- UNKNOWN: efeito E2E definitivo do KDA8, demais candidatos b12x/atenção/drafts, distribuição estocástica e estabilidade/promoção final continuam pendentes.

2026-09-09T03:02:31.597704+00:00
## Retomada explícita pós-bloqueio
- OBSERVED: usuário retomou explicitamente o objetivo bloqueado após pedido que nomeou reproduçãoTCP efêmera, correção reversível launcher/monitor, reinício privado34731 e benchmarks. Grant aplicado somente aessa recuperação, preservando todos limites; goal_id9a3baf15-5975-47c0-b9a3-fa2fce3053a5.
- Status: OPEN; contador de bloqueio reiniciado.
### X3 — controles TCP descartáveis
- Hypothesis: H2/TIME_WAIT consegue produzir exatamente bindErrno98 mesmo sem listener; reutilização documentada resolve somente estado residual, não listener ativo.
- Decision changes: aplicar SO_REUSEADDR no preflight somente se controle provar solução sem aceitar conflito com listener ativo. Causa histórica continuaUNKNOWN.
- Novelty: primeira reprodução determinística, agora explicitamente autorizada; não repetir startup real.
- Exact action: scripts/tcp-control024.py porremote.py, stdlibsocket, bind127.0.0.1:0 somente, três controles, timeout30s.
- Prediction: bindfresco PASS; listenerativo impede segundo bind comreuse; após conexão fechada primeiro pelo servidor, rawbindErrno98 masreusebindPASS.
- Refutes: rawbindPASS enfraquece reproduçãoTIME_WAIT; reuse aceita listenerativo rejeita solução.
- Resource proof: DISPOSABLE_VERIFIED — portas atribuídas porOS a sockets novos desteprocesso, assert fora22/8002/8081/34731/34732; cliente conecta somente endereço recém-alocado. Sem modelos/GPU/arquivos remotos/serviços existentes. Bytes sintéticos, sockets timeout3s, close apenas próprios.
- Can affect: sockets/RAM exclusivos e JSON local da campanha; nenhumkill/reset/configsegurança/cleanup.
- Reversal: fechamento apenas sockets criados pela prova; estado TCP próprio expira peloOS.
- Gates: autorização pós-incidente; decisão discriminante; novo; recursos próprios provados pelaalocação; blast radius exclusivo; efeitos reversíveis independentes; limite30s.
- Priority: prova do mecanismo antes da correção real; repetir snapshot tardio não recuperariahistória.

### X3 result
- OBSERVED E9: evidence/tcp-control024.json: {"address": ["127.0.0.1", 43057], "fresh_bind_errno": 0, "live_listener_reuse_errno": 98, "after_active_close_raw_errno": 98, "after_active_close_reuse_errno": 0, "passed": true}
- Prediction verdict: confirmed; mecanismo reproduzido em socket próprio, não prova retroativa da causa024.
- INFERRED: SO_REUSEADDR admite estado residual próprio sem tomar listenerativo; solução candidata apoiada em controle positivo/negativo. Identidade de processo/boot e ausência de listener continuarão obrigatórias.

X4 pré-registro IMPLEMENT/VERIFY: corrigir scripts/exec-manifest.py (registrar identidade antes dos preflights e SO_REUSEADDR somente socket preflight); convergir scripts/wait-ready024.py para watcher parametrizado pelo launch.json, esperar recibo30s, checar boot/start/zombie, não consultarAPI enquanto processo anteriorvive, retornar1 ao erro/timeout. Sem retries do launcher nem tomar listenerativo. Preservar fontes024 históricas remotas. Novo exp024b-control8 e/tmp/qwen-exp024b, perfil016 idêntico, endpoint34731, mesmos recursos sintéticos/cachecampanha. Validarcontrole watcher comreciboPIDdebootinválido semAPI efail<3s; depois launcherWindows persistente inicia024b, watcherREADY e suitecomum. Semalteração de pesos/kernel/qualidade nesta recuperação. Fonteautoridade: exec-manifest.py e wait-ready024.py; cópias anteriores são evidência histórica, não outros mecanismos ativos. Rollback: não executar fontes novas; nenhuma remoção/reescrita remota de dados antigos.

### X4 result — recuperação parcial, nova falha024B
- OBSERVED: controle watcher com boot sintético inválido retornou1 em0,04022s, sem chamadaAPI; teste negativo PASS. AST das duasfontes PASS e diff focado inspecionado: recibo antecipado, SO_REUSEADDR, watcher fail-fast/identidade/semproxy.
- OBSERVED: controladorWindows24912 iniciou024B; stdout passou bind e chegou stageexec/PID276. Modelo/draft foram carregados, mas NVCC falhou: Could not open output file '/tmp/qwen-exp024b/tmp/tmpxft_...'; diagnóstico evidence/incident024b.json confirmou TMPDIR ausente. Watcher retornou1 imediatamente; benchmark não executado (set-e).
- Prediction verdict: refuted para recuperação completa; confirmed para bind e fail-fast. Não alegar READY nem ganho de desempenho.
- INFERRED: diretórioTMP criado na preparação separada não é pré-requisito durável; falta assegurar existência no mesmo processo persistente antes deexecutar runtime. Boot atual40a3fb15-1ba1-42b7-9008-2d0de794b62c, diferente do anterior; perda de/tmp é compatível com cicloWSL. Instante exato da remoção não foi observado.
- OBSERVED: falha deTMP já havia aparecido na preparaçãoMICRO022B; a correção local daquelecaso não foi incorporada ao launcher. Omissão do orquestrador, não falta de autorização original nem defeito dos pesos.
- Safety: nova falha operacional congelou mutações; logs/fontes/artefatos preservados, sem retry/cleanup. Modelo encerrou por falha do scheduler; não enviamoskill adicional.
- New exact correction proposed, NOT EXECUTED: criar/verificar somente TMPDIR próprio declarado (/tmp/qwen-<run>/tmp), com guard de path/symlink, dentro de exec-manifest.py imediatamente antes deexecve; bloquearfallback e testarcriação quandoausente. Nenhum /tmp global será limpo; nenhuma alteraçãoWsl/segurança/SDK.
- Authorization boundary: pedir aprovação pós-novafalha para essa correção deTMP e nova inicialização isolada, preservando todoslimites. Status BLOCKED pararecuperação; objetivo incompleto. Contador de novo bloqueio começa neste turno, não reutilizar três turnos antigos.

## Continuação8 — diagnóstico permitido, execução ainda congelada
- OBSERVED: continuação automática8 não contém aprovação explícita da nova correção TMPDIR após falha024B; segundo turno com esse bloqueio (retomada anterior+8), sem usar contagem doincidente024 antigo.
- OBSERVED: coleta read-only completou reject_sampling.py e wrappers sgl_kernel/speculative.py/__init__.py/load_utils.py/version.py. Pacote instalado declara0.4.6.post1, metadata dir sglang_kernel-0.4.6.post1.dist-info; binários common_ops.abi3.so existem sobsm100/sm90. Inventário em evidence/sampling-primitive-inventory.json.
- OBSERVED: tree_speculative_sampling_target_only é wrapper para torch.ops.sgl_kernel.tree_speculative_sampling_target_only.default, repassando thresholds1.0; não contém fórmula deaceitação. UNKNOWN: implementação compilada e equivalência estocástica; não promover nomes/wrappers a prova matemática.
- OBSERVED: nenhuma importação de aplicação, testeGPU, correçãoTMP ou reinício nesta continuação. Fontecoletada é evidência, não alteração dopacote instalado.

## Continuação9 — terceiro turno do novo bloqueio
- OBSERVED: retomada explícita anterior, continuação8 e continuação9 constituem três turnos deste objetivo com falha024B/necessidade de autorização da correçãoTMPDIR. Não reutilizar contagem do objetivo anterior.
- OBSERVED: artefato evidence/incident024b.json reconferido localmente: stageexec alcançado, NVCC Could not open output file em/tmp/qwen-exp024b/tmp, tmp_exists=false. Isso não é uma nova observação remota nem prova doestado atual dohost.
- OBSERVED: após pedido pós-falha024B só chegaram continuações automáticas do objetivo original; nenhuma aprovação explícita da correçãoTMPDIR/repetição foi recebida.
- Status: BLOCKED. Ação necessária: autorizar criação/verificação apenas doTMPDIR próprio pelo launcher persistente, testeisolado e reinício privado34731, mantendo fontes/logs anteriores, semcleanup, modelos/SDK/SSH/túnel intocados.
- UNKNOWN: objetivo total permanece incompleto — controle contemporâneo, amostragem, candidatos restantes e promoção/estabilidade final. Nenhuma afirmação de oportunidades esgotadas.

## Solicitação direta: iniciar agora e adicionar ao PI
OBSERVED: usuário autoriza iniciar após falhaTMP e cadastrar modelo no PI. Escopo imediato: melhor referência validada016 (semhookexperimental), normal27B, contexto131200/pool131328, DFlash8; disponibilizar em8081 via túnel existente8002→8081 e registrar provider local sem trocar default. Esta entrega NÃO encerra objetivo amplo deotimização.
OBSERVED: preflight SSH inicialmentePermissionDenied; ssh-agent vazio. UseKeychain=yes comchavejáexistente autenticou, semalterarcredenciais/permissões. remote.py ajustado apenas paraessa leitura suportada. Host semlistener deAPI, GPU30116MiBlivres,460W inalterados.
Pré-registro: exec-manifest.py criará/verificará TMPDIR próprio dentro doprocesso persistente (guard absoluto,/tmp/qwen-namespace, sem.. oulinks, UIDpróprio, curto); nada de/tmp global limpo. Novo pi-serving-025/HOME/cache/storage próprio, dadossintéticos devalidação, modelosSDKsomenteleitura, compilercachecampanha sequencial. Testarhelper criaçãoausente e rejeiçãopathinválido antes deGPU. WindowsCIM mantémWSLforeground como recuperação anterior, porta8081 explicitamente solicitada parauso. VerificarendpointLinux, endpointWindows, túnelMac e integraçãoPI; nãoexecutartools domodelo fora fixture própria. ConfiguraçãoPI autoridade ainda sendo apurada por scout; preservartodosproviders e default.

PI025 progresso: TMPhelper criaçãoausente/rejeição.. PASS; watcherreciboinválido exit1 em0,0385s. WindowsCIM9780 iniciouperfil016 em8081, READY após58s; GPU/SDK/pesos inalterados porconfig. Conta llama-tunnel rejeitou chave desteMac; nenhuma alteração deSSHfeita. Túnel8002→8081 estabelecido comcontagpu-host jáautorizada, /v1/modelsMac confirmouqwen3.8-27b-normal.
ConfigPI: somenteprovider gpu-host adicionado em models.json e config/models.json. Divergência preexistente openai-codex.modelOverrides preservada, não forçar igualdade nem executar smoke globalqueexigeigualdade. models-store/settings/default/extensões inalterados.
Pré-registro smokePI: importar ModelRuntime/ModelRegistry e adapterPiAI instalados; credentials e modelscache exclusivamenteemRAM, PI_OFFLINE1, catálogoexternodesabilitado, fetch guard apenas127.0.0.1:8002/chat/completions semredirect; prompts/ferramentaadd_numbers sintéticos, semexecução de tools reais. Verificarregistro disponível e131072, stream comraciocínio +toolcall17,25 e retorno42. SaídaJSON exclusiva localpi-smoke025; nenhumauth.json lido/escrito nem providerpago solicitado.

### PI025 resultado: serviço disponível para teste
OBSERVED: modelo qwen3.8-27b-normal READY em8081; localhostMac8002 encaminha8081 via SSHcontagpu-host existente. PIDLinux387/boot/startconfirmados apósvárias desconexõesSSH; TMPDIRpróprio existe. Dpkgstatus efontegateKDA hashesinalterados. Não ativadohookKDAexperimental; perfilreferência016 preservado.
OBSERVED: ModelRuntime+ModelRegistryPI instalados carregaramconfigreal e disponibilizaramgpu-host/qwen3.8-27b-normal,contextWindow131072. Doisrequests viaadapterPI passaramtoolcalladd_numbers(17,25), retorno{"sum":42} e raciocínio348chars. Evidence/pi-smoke025.json; nenhuma ferramenta real executada, credentials/modelscacheRAM, nenhuma redeproviderpago.
OBSERVED: provider adicionado somente a ~/.pi/agent/models.json e config/models.json; default mantido. Diferença preexistente de modelOverridesopenai-codex preservada. Diff focado removidochurn deformatoinputarrays. models-store/auth/settings/extensões intocados.
INFERRED: recuperação operacional e integraçãoPI concluídas nesteescopo; TIME_WAIThistórico024 continua semprova retroativa. Otimizaçãoglobal/validaçãoestocástica continuam não concluídas; não chamar goal_complete. Usuário pediu serviço paratestaragora: suspenderbenchmarks/restarts queinterrompam seuuso.

### TOOL026 — instrumentação isolada — 2026-09-09T16:08:14.670759+00:00
OBSERVED: ncu/nsys/perf não estão noPATH; nenhum diretório/opt/nvidia apareceu noescopo consultado. Não é prova de ausência emtodo host. RETRIEVED: NsightCompute docs2026.2.1 suportamGB20x eWSL2/Win11 driver>=525; requerem acesso aos contadores habilitado peloWindows. Não habilitar/alterar essa permissão. Defaultclock-control mudou para boost; qualquercoleta usará explicitamente --clock-control none. Fonte https://docs.nvidia.com/nsight-compute/ReleaseNotes/index.html .
Pré-registro: baixar pacoteoficial nsight-compute-2026.2.1_2026.2.1.5-1_amd64.deb (345070298bytes,SHA2566829651ceeb0c3f65890b9f727b74d1e550fed58c454e11c2c87442295e4eb70) para novo root ncu-tool026 e listarconteúdo. Somente aquisição, semapt/installscript/GUI/execGPU; exigirlivre>=5GiB. Dados/arquivos novospertencem àcampanha, pacote/hashmetadata oficial em evidence/ncu-packages026.json. Não alterarOS/SDK/cachemodelos nem interromperservidor. Limite360sdownload, preservarfalhas. Extrair/executar apenas apósauditar layout/links. Decisão: CLIisoladadisponível versusdependências/acesso aindaUNKNOWN; não ébenchmark.

TOOL026 aquisição resultado: VERIFIED em15,05s, tamanho/hash coincidem comíndiceNVIDIA. Conteúdo sobopt/nvidia/nsight-compute/2026.2.1; links listados são relativos àprópria biblioteca. Nenhum instalador executado.
TOOL026 próximo pré-registro: extrair somente árvoreopt/nvidia/nsight-compute/2026.2.1 em ncu-tool026/root comtarfilefilter=data, recusarpathsfora/linksabsolutos ou especiais, limitar2GiBexpandidos. Dpkg--fsys-tarfile sólêarquivo, nãoexecutahooks. CLI --version/--help somente, HOME/XDG/TMP próprios; sem aplicaçãoalvo, --attach ou coletaGPU, semalteraçãoOS/driver/clocks/permissões. Oracle: versão2026.2.1 ehelp disponível; hashdpkgantes/depoisigual. Stop120s; qualquerfalha preservaárvoreparcial, semcleanup.

TOOL026 extração resultado:2626arquivos/1356568813bytes, CLI2026.2.1.0 build38283040 responde--version/--help; DPKGhashinalterado. Pacoteextraído isoladamente, seminstalaçãoglobal. Coleta decontadores e sua permissão aindaUNKNOWN; nenhumaGPUaplicação instrumentada.

### MICRO027 — contrato de comparação corrigida (antes do código/execuçãoGPU)
OBSERVED: cache real do servidor contém CutlassFp4GemmRunner tática4 paraM8/K17408/N5120 e tática2 paraM8/K5120/N34816. Chave usa alpha escalar(shape()), enquanto MICRO022 criaalpha[1]; diferença deforma também podeimpedir cachehit. disable_flashinfer_autotune=false/skip_ops=null confirmados. Cópia em evidence/server-autotune026.json;metadataFI0.6.18/CUDA13.0/GPU5090. UNKNOWN: cachehit do novo harness atéobservá-lo, não assumir.
HipóteseH27a: KDA~34us é próximodotunedCUTLASS, explicando diferença micro/E2E; H27b: b12x autotunado reduz tráfego/esperas nessasformas além da referência. Comparar formas down/up,seeds17/29/43,M8; CUTLASScomcacheobrigatório,táticaexata, KDA somentedown, b12x explícitoautotunado. MesmosoperandosNvFP4/BF16/scales128x4, alpha escalarigualservidor. Gatefinito/L2<=1e-3/erroelementar<=0,015625max(|ref|,1), bitwise fraction; semrelaxar. Todoscasosantesdetimings;50samplesCUDAgraphsL2flush256MiB intercalados/preheat>=1s, guardtemposválidos. Comparação é micro, não promessaE2E.
Recursosfuturos: novo micro027/root/HOME/CUDA/CuTe/tmp e cópia somente-leitura docachetuning; nenhuma leitura de pesos ou conversas; >=27000MiB livres e servidor pausadoordenadamente antesdeGPU. Builderapenasimplementa local, nãoexecutaGPU/host. Write set scripts/micro_gemm027.py + parametrizehelpers de scripts/micro_gemm022.py preservandodefaults/casos022. Nenhum pacote/modelo/gate/SDK/configPI muda. Fonte022anterior preservada em evidence/micro022-source-before027.py. CLIstatus deerro preservaresultado, semclaimdetimingsequalqueroráculofalhar.

MICRO027 revisão antes deGPU: corrigidos argumentos torch/flashinfer ausentes na chamada dafactory; removido caminhoMac não disponível noWSL, substituído porSHA256pin do snapshot323a571e80018dc50f0c45593caff5240fb6dcd908f94af633b92f8896894586; exigirload_configs=True explicitamente. Defaults022 preservados no diff. AST/arity/--help são sóevidênciaestática; oráculosGPUaindaUNVERIFIED.

MICRO027 janelaGPU pré-registro: nsightprobe próprio1024float32, umaoperação coletada, --clock-control none/--cache-control none/--config-file0/application-only, métricasduration/DRAMread/write/SMthroughput, exportlogroot027 semsobrescrever. NCUlocks seguemTMPDIR documentado próprio; semarquivo temporário deexport default nem rules/GUI. SeERR_NVGPUCTRPERM, registrarUNVERIFIED semmudança desegurança e continuar microtiming; outra falha deprofiler encerra para diagnóstico. Probe180s;micro1200s. Nenhum requestreal é replayed/instrumentado. API025 será pausada graciosamente comPID/start/boot e fila/runningzero reconferidos, microcontroller esperandoantes; sóprocessopróprio éalvo. >=27000MiB livres antesdeGPU. Restore028 preparado comperfil25mesmospesos/flags/alias8081, caches/storage próprios,TMPcriadonoprocessopersistente. Túnel ePIconfigs permanecem. Não executarrestore antes demicroencerrar; verificarresultado/GPU primeiro.

MICRO027 resultado: job20,90s, microexit0/PASS,6casos/15timings, cachematchpresente e oráculosnuméricos passaram. ContadoresNCU recusados ERR_NVGPUCTRPERM; probe sintético computoucorretamente, acessoHW permaneceUNVERIFIED. Não alterarpermissões; alternativasmensuráveis continuam. GPU30151MiBlivres/460W apósprocessoencerrar. Restauração028 da referência seguiráagora conformejanela planejada; análisequantitativa usaJSONLcompleto evidence/micro027-results.jsonl, não tailtruncado.

### CHECKPOINT028 — 2026-09-09T16:42:11.480982+00:00
OBSERVED: micro027 usou cacheCUTLASS comtáticas4(down)/2(up), hash/correspondência dekeyverificadosantes/depois do tuningb12x;9comparaçõesnuméricas bitwise100%, maxerro0. Medianas das3medianas(seeds) emus: downCUTLASS44,256;KDA38,112 (~13,9% menoslatência);b12x64,736 (~46,3% maislatência). UpCUTLASS68,864/b12x68,864; não háganho robusto dopublicb12xautotunado nessasformas. Evidência micro027-summary.json;não comparar diretamente essesus comtraces instrumentadosnemtraduzir13,9% emganhoE2E.
OBSERVED: b12x escolheu down(tile128x128,prefetchfalse) e up(tile64x128,prefetchtrue); seuplano defaultdocumentado down64x128 difere. UNKNOWN: autotuner minimiza custo sobcondiçõesrepresentativas diferentesdasnossas? Varreduratáticas/cachequente-frio pode discriminar, não repetir simplesbackendtoggle.
OBSERVED: NsightCompute2026.2.1 executouprobeexclusivo, mas Windows negoucontadores ERR_NVGPUCTRPERM. Nenhumapermissão ouclockalterado. Essa rota permaneceindisponível soblimitesatuais; não émotivo para declarar hardware saturado.
OBSERVED: metadata de launches nos traces019/023 extraída em kernel-launch-metadata026.json: exemplosCUTLASS384threads/168regs e84–96KiBshared;KDA96threads/98regs,89088bytes shared, grid80blocos. Campoest.occupancy retorna0/SMEMblocklimit0 apesar dekernels executarem. INFERRED: esseestimador não é prova confiável deocupação nesta configuração (possíveltratamento de sharedopt-in UNKNOWN). Não confundir blocks-per-SM daatividade comocupação medida; validar limites reaisCUDA/geometry antesdeconclusões.
OBSERVED: API restaurada empi-serving-028 viaWindowsCIM29704, READY após54,07s, endpointMac8002READY/context131200/pool131328. Túnel/PIconfigs/pesos nãoalterados. Goal3c0032b7-57bc-45d3-bf5c-0904f9ad6266 incompleto: investigação memória/tiles/atenção/drafts, controleE2E atualizado e qualidadeestocástica permanecem. Próximasprovas: limitesCUDA e diferenças deworking-set/táticas, sem contornarrestriçãodecontadores e sempre comGPUexclusiva nosbenchmarks.

READ029: CUDA driver query semcontexto/alocação confirmou170SM,96MiBL2,100KiBshared/SM,99KiBsharedopt-in/bloco,48KiBdefault,65536regs/SM. Limiteanalítico doslaunchesde84–96KiB é1CTA/SM; nãoéocupação medida. MetadataPython: duasconsultas falharam (interpretadorsistema; nomeincorreto sgl-kernel); diretórioscopadomostrou nomecorreto sglang_kernel-0.4.6.post1.dist-info. Terceiraleitura corrigida apenasapósnovaevidência; nenhumpackageexecutado/modificado.

### MICRO029 — táticas explícitas e efeito de reutilização — 2026-09-09T16:54:34.782859+00:00
Hipótese antes da implementação: a seleção b12x de MICRO027 pode não minimizar latência no regime de execução da conversa. A seleção automática usa buffers clonados (3×L2); nosso harness usa flush de 256 MiB. Ambas buscam cache frio, mas não são idênticas. Enumerar as oito táticas STOCK realmente retornadas por B12xFp4GemmRunner; não inventar tiles, alterar matemática ou modificar bibliotecas instaladas.
Contrato: M8, mesmas projeções down/up e seeds17/29/43; referência CUTLASS com cache/hash/key/tática 4/2 exigidos; KDA somente down. Para cada tática, gate numérico inalterado antes de qualquer timing. Comparar regime flush256MiB versus chamada imediatamente precedida por uma chamada idêntica sem flush (reutilização; não declarar que toda a matriz cabe no L2). Mesmos operandos, 50 amostras intercaladas com rotação da ordem e preheat>=1s por regime. Registrar identidade de runner/tática e cache de compilação, todos os valores, hashes e falhas. Comparação sintética, não ganho de conversa.
Write set local: scripts/micro_gemm029.py + parâmetros opcionais de _capture_graphs em micro_gemm022.py com defaults preservados. Backup022 em evidence/micro022-source-before029.py. Nenhuma alteração de servidor/modelo/SDK/PI/segurança; operação remota somente pelo orquestrador. Novo root futuro micro029 e TMP próprio, GPU exclusiva verificada antes da execução; sem pesos, conversas reais ou rede no benchmark. API028 permanece ativa durante implementação.
Fonte de sampling adquirida separadamente em evidence/upstream/kernel029, commit01e19ddf5. Raw HTTP503 Backend.max_conn reached; recuperação via Contents API sequencial, preservando arquivos já obtidos. Wheel instalado se chama sglang-kernel0.4.6.post1; vínculo binário exato com commit ainda UNKNOWN. Não executar testes upstream diretamente.

### SAMPLING031 — pré-registro de prova do binário
A fonte CUDA recuperada faz aceitação U<=q(c) nos thresholds1/1; em rejeição grava p(c)=q(c) e amostra relu(q-p) normalizado. Isso fecha a demonstração da distribuição construída pelo DFlash em aritmética ideal; não prova o binário instalado. Runtime028 confirmado DFLASH/block8/thresholds1.0 e1.0/rejection_sampling=false. O flag false não aparece no worker DFlash inspecionado; o caminho não-greedy chama target-only independentemente desse literal.
Lacuna descoberta: o construtor de probabilidades DFlash aplica temperatura/top-k/top-p, mas não min_p. Não alterar silenciosamente o sampler nem declarar suporte a min_p. Esse comportamento precisa de tratamento separado antes de conclusão global. O contraexemplo do scout com threshold_acc=2 está fora do domínio aceito [0,1] e não será usado; valores abaixo de1 também podem enviesar, por exemplo q(c)=0.2/threshold_acc0.5 produz aceitação0.4.
Contrato SAMPLING031: novo script sintético, sem pesos/modelos, testa o op CUDA instalado com controles determinísticos de aceitação/rejeição e distribuição de primeiro token (quatro categorias, candidatos frequente/raro/zero e três seeds;65536 amostras por fixture; limite6sigma+2/N por categoria). Thresholds1/1, índice global flattened por request, dados mutáveis privados por chamada. Registrar também controle negativo de threshold_acc0.5 para q(c)=0.2: deve evidenciar viés, nunca mudança no servidor. Validação do kernel separada da construção de q e de qualidade semântica.
Write set único scripts/sampling031.py; builder só implementação/offline. Futuro root sampling031/TMP próprio, GPU exclusiva, filho separado, caches exclusivos; nenhum teste upstream executado. Execução remota posterior pelo orquestrador, não paralela ao MICRO029. API028 permanece ativa nesta preparação.
MICRO029 isolamento adicional: todos os caches mutáveis passarão a namespaces novos; copiar apenas .so finais FlashInfer e JSONs de autotune SGLang. Não copiar Ninja/build recipes ou metadata com destinos absolutos antigos. Controles before/after dos caches originais; sem exclusões. Restabelecer referência empi-serving-030 com mesmo perfil/tuning e namespaces próprios após janela planejada.

MICRO029 resultado 2026-09-09T17:19:13.321272+00:00: ERROR antes de qualquer timing/resultado numérico. Reference cache hit4 foi observado, mas o recibo de b12x consultava _B12X_MM_FP4_KERNEL_CACHE no módulo errado. Cache é definido em gemm_base.py:6684 e passado ao helper em6852; gemm_mm_fp4_cute_dsl.py apenas define função que monta a chave. Erro de handoff/revisão, não falha demonstrada do kernel. Corrigir somente o namespace do recibo e repetir em novo run; preservar029. Compilação inicial com cache isolado levou ~380s, total393,7s. Post029 PASS: caches antigos byte-a-byte inalterados, fonteKDA preservada, GPU livre29851MiB/limite460W. Fim normal da janela experimental com resultado negativo de harness; restaurar referência030 preparada, sem alteração de modelos/flags/túnel. Sem claim de desempenho.

### MICRO029b / AOT032 — correção e pré-registro
Referência030 voltou READY em725,94s: caches isolados sem receitas Ninja implicaram recompilação, não lentidão de geração. Isso amplia demais a manutenção. Fonte jit/core.py explica: JitSpecNvcc.try_load só aceita caminho AOT; .so em caminho JIT sempre passa pelo build/Ninja. JitSpecNvcc.load(so_path) existe; env.py documenta override de globals por scripts AOT.
Hipótese: usar o fast path AOT EXISTENTE para cópias novas, hash-verificadas, das mesmas bibliotecas funcionais preserva matemática e evita recompilar. Hook process-local sitecustomize-prebuilt032.py valida root, cadaSHA, FI0.6.18 e arch120f, e configura SOMENTE jit_env.FLASHINFER_AOT_DIR. Não alteraJitSpec nem bibliotecas instaladas, não desliga validação de versão, não altera clocks/permissões. Miss continua JIT emcacheprivado. Manifestos e .so emrootnovo; nenhuma receita antiga copiada. Gate de hash falha fechado comSystemExit; APIflags/modelos continuam iguais.
029b corrige apenas owner do cache b12x para gemm_base; novo root preserva029 falho. Repetir mesma matriz e gates. Comparação de tempos só depois de todos numéricos passarem. AOT é mecanismo de carregamento, não candidato a ganho de geração. Restore032 recebe mesmoAOTcatalog eperfil. Artefatos/condições no prepare029b-local.py; 030permaneceatéfila0 econtroladorvalidado. Na preparação local uma asserção detectou substituição errada de run-id antes de escrita/ação remota; corrigida substituição exata.

029b: AOT carregou9 bibliotecas verificadas, eliminando a compilação observada (job8,03s), mas novo recibo falhou: o helper embrulha a chave em(device_index,cache_key). Fonte gemm_mm_fp4_cute_dsl.py:76-79/116 confirma a transformação final; os dois handoffs omitiram esse passo. Nenhumtiming foi emitido. Post029b PASS, caches030 inalterados/GPU29838MiB/460W. Pausa mantida por uma repetição curta029c com recursos novos;032 já preparado será restaurado ao final. Corrigir o recibo para chave completa(device,tuple), não remover o gate. Nova hipótese fundamentada no helper completo, não repetição cega; resultados anteriores preservados.

### CHECKPOINT032 — resultados válidos e serviço observado
MICRO029c PASS em26,35s: 51 comparações numéricas, todas bitwise100%/L2erro0;114 registros de timing, todas as oito táticas STOCK, duas formas, três seeds e dois regimes. Ref CUTLASS down44,256us; KDA38,112us; melhorb12x64x128/no-prefetch54,160us. Up ref68,832us; melhorb12x68,864us: empate, sem ganho. Nenhuma tática b12x promovida; espaço STOCK encerrado para essasduasformas/regimes. Arquivos micro029c-results.jsonl/micro029c-summary.json/post029c.json.
Reutilização imediata: CUTLASS down21,744us versus44,256us comflush; up38,144us versus68,832us. Evidência de sensibilidade ao regime de memória/cache, não prova de saturação deDRAM/ocupação e não ganhoE2E. Atenção às unidades: pesoFP4+scales up=100270080bytes=95,625MiB, quase o L2 de96MiB, não maior por simples comparação100MB>96MiB. Saída/input/outros dados também competem; não declarar residência total. O workload real percorre muito mais pesos por rodada.
AOT032: referência voltouREADY em72,64s contra725,94s da preparação030 semfastpath. Mesmas bibliotecas funcionais emcópias privadas comhashesverificados; sócarregamento mudou, não matemática. Caches protegidos030 byte-a-byte inalterados após029b/029c, fonteKDAintacta, limite460W. ConsultaSSH posterior confirmoumesmaidentidadePID/boot/start, APIready, contexto131200/pool131328; túnelMac8002modelpass. Evidência recon032.json. Nenhuma promoção dekernel/draft; referência block8 continua funcionando.
SAMPLING031 builder entregou scripts/sampling031.py (307linhas), AST/help/arity14 offline relatados; ainda requer revisãoorquestrador eGPU emrunisolado. Ferramenta de agentes não encontrouID após finalização; resultado recuperado do registro JSONL final (sem nova execução). Próximas obrigações: executar prova dobinário, tratar min_p comcontrato preservador, explorar apenasnovas hipóteses kernel/layout/atenção apoiadasnasmedidas, realizarcontroleE2E+contexto+qualidade+tools antes de promoção/finalização. Goal permaneceincompleto.

### SAMPLING031 — revisão e admissão de execução
Revisão direta do wrapper e .cu confirmou ordem de14argumentos, tipos, dimensões e domínio dos thresholds. Corrigido harness antes de GPU: três seeds agora cruzadas comtrêsfixtures (9estatísticos, não só3); identificação de bibliotecas inclui caminho/sgl_kernel/ mesmo que módulo se chamecommon_ops; falhas estatísticas preservam contagens/tolerâncias. Oráculo inalterado. AST/help offline; nenhumresultadoGPU inferido.
Execução: novo sampling031, filho comtimeout240s, fixtures privadas, zero pesos/conversas/rede/cleanup; todos caches eTMP próprios, catálogoAOT de cópias verificadas. Primeiro métricas fila/running0; controlar PID/boot/start antesSIGTERM e aguardar liberaçãoGPU>=27000MiB. Postcheck de caches32 e fonteKDA; fim da janela planejada comrestauração033 mesmo perfil/contexto/modelo/túnel. Falhas de oráculo são resultados negativos preservados, nunca motivo para mudar expectativa.

SAMPLING031 resultado: PASS em3,13s; controles determinísticos aceitação/rejeição e9fixtures estatísticos(65536amostras cada) passaram. Controle negativo confirmou0,401108 versusqoriginal0,2 comthreshold_acc0,5, como previsto; servidorpermanece1/1. Post031 PASS: caches32inalterados, KDAhashpreservado,29838MiBlivres/460W. Evidências sampling031-results.jsonl/post031.json; prova do op fornecidoq, não do min_p nem de todaqualidadesemântica. Restauração033 segueconformeplanejado.

### MICRO034 — próximo experimento fundamentado
Todas táticas stock perderam/empataram. A fonte SM120 permite explicitamente swap_ab=True com tilesN16/32, mas o autotuner não os enumera para M8/Ngrande. Hipótese: reduzir padding da dimensão pequena e mudar orientação dosoperandos melhorauso do kernel além dos tilesstock. Testar somente quatro táticas(64,16)/(64,32), prefetchFalse/True, swapTrue; mantercontrole b12x64x128no-prefetch, CUTLASSmatched eKDA(down). Não combinar split-K oucpasync nesta hipótese.
Fonte can_implement confirma elegibilidadeestática, não correção/runtime. Runnerjápassaswap internamente e preservaCrow-major; ownercachegemm_base comchave(device,cache_key) agora conhecido até armazenamento. Reusar UM harness029 comperfil034 pequeno e gate explícito para não-stock; preservarperfilstock029, todos gates/tempos/seeds anteriores. Novo helper de admissão separado evita fingir que o autotuner enumerou as táticasnovas. Numerics antes dequalquer timing; no ganho E2E inferido.
Write set: scripts/micro_gemm029.py (pontos de perfil/validador e metadados, defaultsintactos), scripts/micro_gemm034.py (perfil). Backup029-before034 preservado. Builder sócódigo/offline. Run034/TMP/caches novos serão preparados quandoharness revisado; API033 ficaativa durantepreparação. Split-K existeparcialmente mas swap+split não temcontratoestabelecido; permaneceDEFER, sem improvisaratomics/precisão.

### MINP035 — contrato reconciliado (sem promoção ainda)
Fonte confirma min_p default0, nenhuma aplicação no target-only DFlash, mas sampler ordinário faz temperatura→topk→topp→minp. Mantém prob>=maxprob*minp (empate sobrevive). O builderDFlash retornaTensorDENSO[bs,draft_tokens,vocab] também no caminho topk otimizado; índices são scatterados antes do retorno. compute_dflash_sampling_correct_drafts_and_bonus resolve o builder pelo globaldflash_utils; patch desse global alcança aliases do worker.
Mudança pretendida: helperprocess-local que envolve APENAS o builder deq. min_p0 retornaobjetooriginal sem operações extras deTensor; minp>0 filtra após topk/topp e normaliza q antes do target-only (que exige distribuição). Batchmisto mantém linhasminp0 bitwise intactas. Sem alterar thresholds1/1, draft, pesos, parser, default ou samplerordinário. Não usar alteração para mascarar regressão. Novo código sóserá carregado emservidor após testes e integração observada.
Write set scripts/dflash_minp035.py + scripts/test_dflash_minp035.py. Testes privados de distribuição direta/empate, mixedbatch, caminhos topk sparse/dense, minp1 e minp0 identidade, instalação idempotente do global correto. Não exercitar endpoints/recursospreexistentes pelo builder; GPU emnovo run pelo orquestrador. API033 permanece funcionando. Lacuna do caminho alternativo selector/accept_sampling não será declarada corrigida semevidência; contrato desta mudança étarget-only usado pelo perfil de referência.

ROOFLINE034: páginaoficialNVIDIA especifica1792GB/s nominais (FE/referência, não medidonogpu-host), https://www.nvidia.com/en-us/geforce/graphics-cards/50-series/rtx-5090/ . Apenasler umavez pesosFP4+SF16 implicaria pisosotimistas~27,98us down,55,95us up e399,09us LMhead. Não écontadorDRAM nemprova de cadaacesso; caches/duplicações/ativação/ocupação mudamtempo. Profiling019deGPU kernels(sem comparartimerAPI) mostrouup~62,9us eLMhead~603,6us; hárazão paraexaminar LMhead separadamente, não inferir ganhosenormes noMLPquase-limite. Modelo realconfirmaVocab248320/hidden5120/64layers em model-shapes034.json. Potênciapreservada460W, nenhumaalteração a575WnominaisdaFE.

ADMISSION034/035: diff029 mantém defaults/gates/tempos; perfil034 declara5táticas, valida stock real ecan_implement narrow separadamente, todasprefixosb12x passamnumérico antesdetiming, fullcachekey(device,tuple)preservado. Helperminp035 lidointegralmente; testes corrigidos paraVocab5/maxTopK4 realmenteentrar sparse, spydohelpertopk confirma ramificação. Revisão mudou fixturedeempate paraGPU eadicionouexpectedsegundaposiçãodedraft. Helperassumeqválido eintervalominpvalidado porSamplingParams; não adiciona clipping/fallbacksilencioso.
Janela planejada: executar034 e035 SERIALMENTE emroots separados, cadaqualcomcontextoGPU/caches/TMP próprios;035 sóinicia após034 encerrar epostcheckconfirmarGPUlivre/protegidosinalterados. 036restaura referência ao final, aindaSEM patchminp atéprova de integraçãodomodelo. Nunca rodarambosemparalelo. AST/helpdeambospassaram; resultadosGPU aindaUNKNOWN antesdajanela.

CHECKPOINT036: MICRO034 PASS33comparaçõesbitwise100%/78timings,masnarrow-swap perdeu; ver micro034-summary.json. MINP035 passoufixturesempateGPU,mixedbatch,zeroidentidade ebuildersdense/sparse reais (spy2/0 confirma osramos), minp1ties. Patch aindaNÃO carregado noserviço, aguarda integraçãoreal. Post034/035caches33inalterados/KDAhashpreservado/460W; referência036restauradaREADYem60,09s. Goalcontinua: integraçãominp,LMhead/KDA eatenção/split-K conformecontratosmedidos, controleE2E+contexto/quality/tools econfigfinal.

### Continuação14 — reconciliação036 / integraçãoMINP038 (antes de escrever/executar)
READ036: PID363/boot/start conferidos; /get_server_info está flat (primeiroprobe assumiu server_args aninhado e falhou KeyError, semefeito); READY/context131200/pool131328/block8/460W confirmados emrecon036.json. Nenhum incidenteoperacional observado. Prioridade: LMhead (scout limitadoemcurso), integração minp, depoisatenção/splitK conformeprovas. Outra rodadaMLPstock/narrow semnovahipótese nãoadmitida.
H38: carregar helper035 por sitecustomize process-local apósAOT, pin deSHA dafonteoriginal+helper, preservaráfastpath0 e aplicaráminp positivo no modelo real. Primeiro teste emnovo exp038-minp/porta34731/aliasqwen-normal-exp038, caches/HOME/TMP/storage novos, modelos/SDK read-only. Registro de primeira chamada positiva sómetadados deforma, sem valores/conversas. Gate: installed callable/hash/marker, minp0 retornooriginal, requestsreaiscomminp0.1 e0.7temperatura, aritmética/raciocínio/toolcall+resultado42, 3seeds de aritméticasimples; comparação minp1 comgreedy comoobservação(nãoexigirigualdade se máximosempatados). Nenhuma promoção viahashsozinho. 27casos/contextoE2E seguirão antesdefinalizar.
Write set delegado: scripts/sitecustomize-minp038.py e scripts/quality-minp038.py apenas. Builder nãoexecutarárede/GPU/server; parent executa janelaexclusiva maisrestauração039. Hipótese deLMhead37 aguarda evidênciadocache/callables, sem mutação antecipada.

### MICRO037 — hipótese LMhead antes da implementação
Novo cache036/digest e7f26be161a1d862 contém headpackedM8/K2560/N248320, CUTLASS tática6; snapshot026 deoutrodigest nãoocontinha. Primeiraglobalscachebuscaapenasnivel1nãolistouarquivos; leituraescopada recursiva esclareceualayout. Snapshot head-server-autotune036.json SHA256678beee4fe0b33f95a2613bf1b7ff3c3705e68e8faad114836377c09fdd6ec68. LogitsProcessor→ModelOptFP4.apply usa GEMMout BF16 e sódepois converte FP32; benchmarkdeBF16 correto paraGEMM, nãooráculo definaislogitsdomodelo.
H37: headpesos+SF~682,03MiB (muitomaiorqueL2), menoslimitado porresidência queMLPup; compararCUTLASS6 às8táticasstockb12x eprivateKDA existente, M8/K5120/N248320,seeds17/29/43, mesmosgates/tempos/cachesde029. PublicKDAgate NÃO suportahead; chamada direta sintética permitida, semalterar whitelist e semclaimsuporteproduto. can_implementb12x aceitaN/K eKDAinterno symbolicNnãoimpõeNfixo; runtimeUNKNOWN. Nenhumkernelnovo/escolhavelocidade comqualidademenor.
Mudança: perfilmicro_gemm037.py; em029 somente elegibilidade/nomeKDA viaumdictperfilável (defaultdowninalterado) ecaveatcacheperfilável. O gate de cache usa globals emmicro_gemm027, portanto perfil deveconfigurar o MAPA EFETIVO também, nãoapenascopiasombra em029. Numericalgate/loops/samples50/flush256MiB/overlap remainunchanged. Backupmicro029-before037 preservado. RunGPU037 adiado atéharnessrevisado/recursospróprios; integração038 podeexecutarprimeiro, restauração039preparada.

### CHECKPOINT040 — MINP038 e LMhead037
MINP038 REAL PASS: privado exp038 carregou AOT+helperhash/dflashsourcehash e wrapper no workerPID881 descendente doprocessoprincipal824. 4aritméticas, raciocínio, toolcalladd_numbers(17,25)+resultado42,3seeds e observaçõesminp0/minp1 sempensamento passaram; 16registros, marcadorpositivo[1,8,248320] apareceu APÓS início dosrequests, comprovandoexecução real. Loggingporinvocação ésomenteinstrumentação privada, semclaimdetempo e nãoirá àconfigfinal. Caches/fontesinstaladasinalterados/29863MiB/460W; privado encerrado. Referência039 depois040 restaurada e túnelMacmodelpass. Decisão: helper semanticamenteadmitido; promoção final depende deconsolidarcom melhorperfil e provar defaultminp0 semregressão.
MICRO037 PASS em20,84s:27comparações(8b12x+privateKDA x3seeds) bitwise100%,60timings. Mediana dasmedianas us L2cold: CUTLASS6 451,808; KDAhead441,840 (+2,21% micro); melhorb12x64x128/no-prefetch455,936 (−0,91%). Reuse: CUTLASS414,944;KDA410,848(+0,99%);b12x64x128414,976(empate). Não promover b12x. KDA head éganho pequeno comoráculo forte; testar combinadoE2E só se atenção não dominar oportunidade maior. PublicKDA gate permaneceinalterado; benchmark privado nãoéclaimde suporte. Caches/fontesinalterados,460W; referência040READY64,49s/contextopendente de reconfinal.
ATTN041 hipótese em investigação: trace longo mostra BatchPrefillWithPagedKV mask0 FP8 ~996ms/16launches. Flag documentado speculative_attention_mode decode e backendsplit podem rotear TARGET_VERIFY ao decode backend; fonte/compatibilidadeTRTLLMMHA+page64 sendo reconciliada antesdequalquerlauncher. Semrodar apenas flagdecode comomesmobackend, pois evidência atual indica HybridAttnBackend só existequando backends diferem e nãohaveria causalidade clara.

### EXP041 — atenção split FlashInfer/TRTLLM-MHA (antes da execução)
H41: o gargalo longo target-verify BatchPrefillWithPaged (~996ms/16launches notrace) pode cair ao usar backendDECODE TRTLLMMHA/XQA apenas para target verify e decode, mantendo ordinaryprefill FlashInfer edraftoverrideFlashInfer. Config adicional explícita: decode-attention-backend trtllm_mha,page-size64,speculative-attention-mode decode,speculative-draft-attention-backend flashinfer. MantémDFlash8/topk1,thresholds1/1,FP8KV,context131200/pool131328,CUTLASSGEMM,SBOverlapping. 131328 e131200 são múltiplosde64.
Fonte confirma HybridAttnBackend target_verify→decode quando modedecode; QwenHybridLinear preserva GDN/full layers; TRTLLM implementa verifydecode+CUDAGraph, exige topk<=1,page16/32/64/128 e bootgate deKVaccess. Warning experimental/unstable exige qualidade/contexto completos; qualquerbootgate/erro/intermitência rejeita. Trial privado exp041 porta34731/aliaspróprio; draftoverrideFI isola alvo. Common suite3needles+27performance e4math+tool obrigatórios. Primeiro comparar exp016/currenthistorical sócomo screening; ganho promissor exigecontrolecontemporâneo. Nenhuma promoção antesdasprovas. Restore042 pré-preparado; roots/caches/HOME/TMP/storage próprios, pesos/SDK/commoncases read-only/copied, fila0 antesSIGTERM, 460W.

EXP041 RESULTADO: BOOTFAIL apósload/pools ao FlashInfer-autotune dummy TARGET_VERIFY entrarTRTLLM XQA semmask; AssertionError Mask is required for speculative decoding. Effective logs provaram split/decode/preffill e draftoverride conformeplanejado; incompatibilidade é do dummyautotune nesta combinação, nãoresultado de desempenho. GPUliberada29863MiB/460W; falhapreservada incident041.json, referência042READY76,05s/túnelpass. Não executar041 novamente.
H43: should_run_flashinfer_autotune fonte mostra flagdisable pula somenteautotune dummy quefalhou. Carregar ANTES oscacheJSONjávalidados viaAutoTuner.load_configs mantém táticasCUTLASS conhecidas, enquanto --disable-flashinfer-autotune evita esse dummy. Hookprivado hash-verifica cadaJSON eAOT, usa APIexistente, failclosed. Previsão: bootseguirá atéCG normal, que forneceSpecInput/mask; seXQA repetir maskausente emCG/request, candidatoincompatível e encerra rota sem patchdebackend. Exp043 novo root, nenhuma mutação/reuso darunfalha. SeREADY, suitecompleta; semclaimdetempo dohook. Restore044 preparado.

EXP043 RESULTADO: BOOTFAIL diferente e conclusivo. CacheAOT+70configsfoipreloaded; autotune dummy foi evitado, mas CUDA-graph targetverify real também chegouXQA semmask e falhou mesmaAssertion Mask is required. Portanto splitTRTLLMMHA+decode é incompatível comDFlash8 nesta versão, não repetir nem patcharframework. Falha preservada incident043.json; GPU29863MiB/460W; referência044READY48,05s/túnelpass.
### EXP045 — split FlashInfer prefill / Triton decode (antes da execução)
H45: TritonAttnBackend implementa explicitamente buffers/mask TARGET_VERIFY, CG e forward_extend comcustommask; split config decode-attention-backend triton + speculative-attention-mode decode + draftoverrideflashinfer mantémordinaryprefill/draft FI e testa hot path long. Pagepermanece1,FP8KV/context/pool/DFlash8/thresholds1/1 mantidos. Previsão: boot passa e long130k pode reduzir BatchPrefillFI; aceitar como melhor geral somente suite27+needles3+math/tools, controlecontemporâneo se screening promissor. triton_attention_num_kv_splits continua default8, semvarrerantesdeprova. Novo exp045/root/cache/TMP/storage privados; restore046 preparado; sempatches/bibliotecas/modelos.

EXP045 RESULTADO: servidorREADY/configefetivaFI-prefill/Triton-decode/modedecode/draftFI/context131200/pool131328. Needles HTTP3/3 eperformance27/27 completaram, mas velocidade foi catastroficamente inferior: medianas tok/s code/math/prose 2k=216,63/179,05/111,88;32k=63,67/62,62/36,04;130k=16,71/17,21/9,99 versus referência016~259,52/228,69/162,07;180,80/225,31/139,41;217,63/203,66/127,81. Quality product/gcd passaram, probability falhou e gate interrompeu antesdoferramentas; candidato REFUTED, não promover nemvarrer Tritonsplits semmicroprova causal diferente. Resultados completos evidence/performance045.jsonl,needles045.jsonl,quality045.jsonl. Unstuck invocado explicitamente após duasfalhasTRTLLM+regressãoTriton; novo caderno .pi/recovery/unstuck/20260909T194255Z-attention-path.md congela toggles e seleciona aquisição source-read-only antesdeoutroprobe. Rollback046 pre-registrado.
ROLLBACK046: exp045 rejeitadopreservado,27performance completos+falhaprobabilityreconfirmados; fila0/PIDpinado, caches/fontesidênticos, GPU29850MiB/460W. pi-serving-046 iniciado porlauncherWindows, READY57,03s; endpoint/túnel normal,FI/prefill/block8/context131200/pool131328. UnstuckX1finalizado; X2source-read-only mapeia fixedsplit/scheduler semnovaGPU.
UNSTUCK X2-X4: source confirmaTARGET_VERIFY FlashInfer não recebefixed_split_size em eager nemCG;None viraauto−1. Determinismo só define tiles paraordinaryextend/decode e tambémdesligaautotune/forçaTorchsampling/2GiBworkspace etc.; rotafixedsplit REFUTED nesta versão. stream_interval sócontrolaOutputStreamer.accept após tokens existirem. Varreduraglobal da instalação encontrou num_continuous_decode_steps somente nadeclaração server_args: campo semconsumidor; ambos fechadoscomo otimizaçãodecompute. Evidence continuous-steps-global046.json e recovery notebook.
UNSTUCK X5 preregistrado: parear exp047-control vs exp048-kda2, mesmasconfigs/seeds/cache/AOT/cases; hook048 hash-pina gate e adiciona SOMENTE down(8,17408,5120)+head(8,5120,248320), ambosbitwisevalidados. 27performance cada; candidato precisa geomeanpositivo/maioriacélulas/nenhumaregressão>10%, depoisneedles+math/tool. Roots/TMP/caches/porta34731 privados e serializados; restore049 preparado; nenhumcleanup/installedsource/model/PI alterado.
UNSTUCK X5/X6 RESULTADO: controle047 vsKDA2-048, 27/27 cada: candidato venceu9/9células, geomean+1,7421%, faixa+0,95..+2,41%, semregressão. Needles3 retornaramJSONexato;4math+raciocínio+tool roundtrippass;markershook presentes. Controleposterior050 confirmouordem reversa: candidato venceu9/9, geomean+2,7542%, faixa+1,56..+3,93%. Hdrift REFUTED, KDA2 SUPPORTED. Doiserroslocais preservados: primeira invocaçãorunner semargs exit2/zerorequests; análise seguinte nãofiltroublankline eJSONDecodeError/semhostmutation. Evidência pair047048050-summary.json,performance047/048/050,quality/needles048.
FINAL053 preregistrado: convergir KDA2+helperminp035+AOT emum único sitecustomize process-local hash-pinado; ummarkerprimeirominp positivo seauto-remove doglobal, então performance medeapenashelperfinalsemobservador. Novo root053/porta8081/aliasnormal espera controle050; restore052 independente. Gates: qualityminp0.1+3seeds+4math+tool,needle3exato,performance27, geomean>amboscontroles/semregressão>10% e dentro2% deKDA048, PI smoke, contexto/pool, PID/boot/start/túnel, caches/fontes/460W e estabilidade>=120s. Qualquerfalha restaura052; semcleanup.
FINAL053 PARCIAL: READY44,04s; qualityfinal combinado PASS (math/reasoning/tool/3seedsminp0.1, markerworker observer_removed=true). Antesdeneedles/performance, harness recusouporta8081 porguard intencional `--url ports 8081 and 8002 are forbidden`; zerocasosbenchmark enviados. Não afrouxarguard nemusarproxy. Serviço053 íntegro/ativo. UnstuckX7 INCONCLUSIVE sóparaperformance.
X8 preregistrado: cloneIDÊNTICO exp054-final-validation em34731 comquality→observerremovido→needles3→performance27; comparar controles047/050 eKDA048. SePASS, promover pi-serving-055 idêntico8081 e repetirquality+PI/túnel/PID/contexto/caches/460W/estabilidade120s. Baseline052 independente sequalquerfalha. benchmark_stream.py intocado.
X8 RESULTADO: clonefinal054 READY60,75s, qualityfinal/markerobserverremoved/needles3exatos/performance27 PASS funcionalmente. MAS perfil combinado KDA2+minp REFUTED para performance: vscontrole047 geomean−1,4174%,5/9melhores,code130−17,12%; vscontrole050 geomean−0,4367%,5/9,code130−16,15%; vsKDA048 geomean−3,1054%, code130−18,47%, fora±2%. Não atribuir automaticamente aominp:ordem/estado também mudaram; decisão independe dacausa porque gate>10% falhou. pi-serving-055 preparadoinerte NÃO seráiniciado. Conforme prereg, restaurarbaseline052. KDA2 isolado continuaSUPPORTED porX5/X6; combinação minp não serápromoção default.
ROLLBACK052 apósX8: 054 parou comfull/hashcheck, GPU29843MiB/460W; baseline052 READY52,73s/túnelpass. 055 permaneceinerte/nãoiniciado, semcleanup. Decisão final: minp correction NÃO promovida no default devido gateperformance refutado; comportamento default existente preservado, caveat minppositivo documentado. KDA2 isolado permanece vencedor comprovado.
X9 preregistrado: pi-serving-056 KDA2-only replica exp048 em8081/aliasnormal; hookhash-pinado+AOT apenas doisshapes, semhelperminp. Restore057baseline independente. Validação final usa privateperformance048/controles047+050/needles+quality, e no processo056 math/reasoning/tool+PI smoke+config/context/pool/túnel/PID/boot/start/caches/fontes/460W+estabilidade>=120s. Falha restaura057.
FINAL056 validação intermediária: READY61,82s. Primeira quality encerrou ANTES de requests: prepare056 omitiu benchmark_stream.py importado porquality-api056, ModuleNotFoundError. Não éfalhamodelo. Nova evidência admite umacorreção: criar exclusivamente snapshotlocal hash-verificado no root056 e repetiruma vez; preservar falha. PI smoke independente jáPASS: provider/model/context131072,2requests,toolroundtrip,pensamento348chars,sum42. Reconread-only também retornounãozero ainda semcausa; diagnosticar antesdeclaim. X9 continuaUNRESOLVED.
FINAL056 RESULTADO: KDA2-only READY61,82s. Quality primeira tentativa falhou antesderequest porbenchmark_stream omitido; AM1 criouexclusivamente snapshot localSHAf9c3... no root, repetiçãoúnica passou4math/raciocínio/tool. PI smokePASS2requests/context131072/thinking348/sum42. Recon launch+hook semanticamenteidênticos ao048,FI/DFLASH8/thresholds1/1/context131200/pool131328,4markersKDA,minpnãoinstalado,caches/fontesintactos,460W. PID5232/start1000310/boot estável125,02s; Windowscontroller28548 vivo/rootmatch. Serviço056 éautoridade atual. Restore057 permaneceinerte.
X10 cpasync prereg: único GEMMdominante aindaCUTLASS éup M8/K5120/N34816. Testar directb12x load_pathcpasync tile64x128/prefetchoff/on vsCUTLASStática2+TMActrl, numerics3seeds antesde50samples cold/reuse; chave própriatodosstaticparams. Exigir>=2% ambosregimes; umrun058/restore059. Split-K sódepois e separado. Serviço056 permanece enquanto códigooffline.
MICRO058 RESULTADO: ERROR antesdetiming. CUTLASScachetática2+doisTMAcontrols compilaram/executaramseed17. Primeirocpasync compile CuTe ICE: pointerFP4 apósrecast alinhamento8bits versus128bits exigidos porsimt_async_copy. FakeUint8 jáassumed_align32; can_implementtrue não cobrecontratoruntime. Semnumericcpasync/semclaimperf. Post caches/KDA intactos/GPU28084MiB/460W; KDAfinal restaurado pi-serving-059 READY110,12s/túnelpass. Não repetir058.
UNSTUCK X11 prereg: buscar source-read-only naCuTe instalada primitive/parâmetro para preservar/provar>=16-byte apósrecast_ptr. Seausente/compilador, cpasync fecha; se APIexplícita, no máximo umpatchemcópiaprivada+novo run. Semimports/GPU.
UNSTUCK X11: primeira busca assumiu pacotecutlass top-level e falhou semefeito; .pth localizou nvidia_cutlass_dsl/dsl_packages/cutlass. recast_ptr não aceitaalign; docs IR dizemrecast não pode converteralign attribute; nenhumassume_aligned. Cpasync fechado atéupstream/versãonova. Split-K tambémnãoadmitido: up jálança272CTAs=1,6/SM; direct-only, publiccachekey omiteparams, non-atomic requerC3D enquanto compilehelperé2D, atomicBF16 exigeinit/redução semintegração pública. Baixovalordedecisão+resource/reversibilityUNKNOWN; não fabricarpatch.
X12 final prereg: verificar processo REAL restante pi-serving-059 (KDA2), não056 histórico. Stage exclusivo benchmark_stream+quality harness hash-verificados, qualitymath/tool, PI smoke059; config/hook equivalência048, context/pool,túnel,PIDidade>=120s,Windowscontroller,fontes/460W. Semperformance repetida; usar controles047/050 e candidato048 jáfechados.
### RESOLUÇÃO FINAL NORMAL — pi-serving-059
OBSERVED: KDA2-only é melhor perfil validado. Candidato048 venceu controleanterior047 9/9células, geomean+1,7421%(faixa+0,95..+2,41%) e controleposterior050 9/9, geomean+2,7542%(+1,56..+3,93%). Medianas candidato tok/s code/math/prose:2k303,14/232,24/167,67;32k235,47/232,05/143,96;130k228,35/181,30/127,01. Perf27/27, completions1024; needles3/3 JSONexato atéprompt130084/total130401 eperformance total131110, preservando>=131072. Quality4math+raciocínio+toolroundtrip PASS; PI059 PASS2requests/contextWindow131072/thinking285/sum42; stochastic final temperature0.7 seeds17/29/43 PASS produto121401+raciocínio.
Serviço deixado: /home/operator/qwen38-5090/optimization-runs/pi-serving-059, modelo qwen3.8-27b-normal, porta8081/túnelMac8002, SGLang0.5.19.dev992, DFlash2 block8 thresholds1/1, context131200,pool131328,FlashInfer/CUTLASS mais KDA somente shapes(8,17408,5120)e(8,5120,248320), AOT9libs hash-verificadas. Hook/launch semanticamenteidênticos ao candidato048. PID5828/start1210775/boot econtrollerWindows36008 persistentes; idade>345s; memória estável125s31032→31029MiBusados(free1156→1159), únicaárvore /dev/dxg é059; powerlimit460W; fontesinstaladasinalteradas; minphelpernãoinstalado.
REFUTED/fechado: NInfer8−8,73%;NInfer4−3,89%;MTP3−30,92%;cuDNNFP4−24,37%;DFlashblocks4/6/9;KDA-downsozinho insuficiente;b12xstock/narrow;TRTLLM-XQA mask incompatível;Triton attention regressão extrema+qualityfail;fixedsplit nãoalcançaTARGET_VERIFY;continuoussteps campomorto;streaminterval apenasemissão;KDA2+minp geomean<=controle/code130−16..18%;cpasync DSL alignmentICE semAPI align;splitK nãoadmitido porgridjá272CTAs econtratosoutput/reduction/integration incompletos. NCU counters continuamERR_NVGPUCTRPERM; segurança nãoalterada. Abliterated continua deliberadamente depois.
RESOLVED: INCIDENT-006, persistência/API/PI, comparações, profiling, otimizações profundas, performance/correção/contexto/quality/reasoning/tools e estado final concluídos. Caderno unstuck resolvido em .pi/recovery/unstuck/20260909T194255Z-attention-path.md. Nenhuma exclusão,gasto,overclock,mudançadeSSH/segurança ou alteração de pacote/modelo foi feita.
DOCUMENTAÇÃO OPERACIONAL FINAL: criado FINAL_RUNBOOK.md com explicação simples, configuração exata, evidências, localização dos artefatos, health check e procedimento seguro de réplica/recovery por novo root versionado. Criado scripts/check-final059.py somente-leitura; valida túnel8002→8081, PID/boot/start, modelo/context131200/pool131328, DFlash8, CUTLASS, hook KDA2, AOT9 hashes e460W. Validação imediata PASS no processo059 PID5828. spawn059.ps1/root059 são single-use e não devem ser reutilizados; após reboot/falha, preservar059 e criar novo root isolado.
QWENCTL WINDOWS APP: implementado Go stdlib+managerPythonembutido, CLI start/stop/status/health/config/logs/version, instalador user-local `%LOCALAPPDATA%\Programs\QwenCtl`,PATH usuário único, semservice/autostart/admin/túnel/security changes. Source/release em windows/qwenctl; build PE32+AMD64 SHA0331c0585642e8b16ddb8e7c77c482c8693b143ef963465df3f68b126184b051. Unit:GoPASS/vetPASS/Python10PASS. Validator final commit1e3c4df PASS. Instalador idempotente, config isolada VALIDATED→CUSTOM_UNVALIDATED→invalidno-write→resetPASS, logsPASS.
QWENCTL LIVE: primeira health detectou falso-negativo root059/KDAsource por template-reference impossível e venv Python symlink resolvido fora; servidor permaneceu intacto. Após confirmação fresca, correções limitadas foram aplicadas/reinstaladas: health HEALTHY. Ciclo real autorizado: stop root059 PID5828 com únicoSIGTERM apósidentity/controller/idle; statusSTOPPED; start criou root isolado `/home/operator/qwen38-5090/optimization-runs/qwenctl-20260910T012754Z-a3c64c11`; READY,PID6820,controllerWindows37328; healthVALIDATED,context131200,pool131328,DFlash8,AOT9,KDAhash,460W. PI smoke pós-restart PASS2requests/toolroundtrip/thinking348/sum42; idade final>479s,idle. Template059 preservou launchSHA db3cac58... e hookSHA11b5044a.... Evidência evidence/qwenctl-final.json,qwenctl-persistence.json,pi-smoke-qwenctl.json.
GITHUB: repositório privado windows/qwenctl criado; main/commit `1e3c4df93b97c94b5cf1b8a6674cc8b452d24d34` enviado e verificado remotamente. README afirma 303,14tok/s como medição do caso curto, não garantia universal.
### EXP-060 — DSpark NVFP4 vs DFlash2
PREREG 2026-09-10: testar draft `gittensor-model-hub/Qwen3.8-27B-DSpark-NVFP4` revisão imutável `eba1ac5a66c74902eaa95a4000a7c5eda96d8e95`, safetensors SHA256 `212fd1b8b5477536ab9e726a94d8565a2246467d044de772f6648df17d5dda05`/1399670058bytes. Modelo card declara mixed NVFP4+BF16, gamma7, target aux layers[4,16,28,40,52], SGLang qwen38 e RTX5090, mas contexto publicado estável ~116K+, abaixo do gate local>=131072, e target certificado não é por identidade o LMHead4 local. H60: compatibilidade local + menor draft podem manter context131200/pool>=131328 e superar DFlash2 block8; refuta se source/config incompatível, boot/131k/quality falhar ou suite E2E não vencer controle contemporâneo. Download primeiro em diretório novo pinado, sem parar serviço, verificar tamanho/hash/config/licença; nenhum cache/modelo existente alterado. Janela GPU somente após artefato pronto: stop via qwenctl identity+idle; root/TMP/cache privado, localhost, mesmos target/CUTLASS/KDA/context/quality/perf. Sem SPS/STS tuning antes de baseline. Sempre restaurar qwenctl DFlash VALIDATED ao final; não promover DSpark sem 27cases+needles+math/tools e ganho uniforme. Nenhuma exclusão/retry em falha; preservar outputs.
