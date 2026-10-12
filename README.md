# Qwen 27B · 300+ tok/s on one RTX 5090

**303.14 tok/s median estimated streaming throughput** in the short coding benchmark: **2,086 input tokens, 1,024 output tokens, three repetitions**, on one **RTX 5090 (32 GB)**. The fastest recorded run in the final candidate suite was **346.93 tok/s** by the same estimate. Including initial waiting and the complete request, the coding case's **median end-to-end throughput was 261.31 tok/s**. These are scenario-specific results, not a 300+ tok/s guarantee for every task or context length. [Recorded evidence](evidence/performance048.jsonl).

Historical research on serving **Qwen3.8-27B NVFP4** with **SGLang, FlashInfer and DFlash2**. Includes the optimization notebook, experimental scripts, recorded benchmarks, the final selective-KDA hook and Windows/WSL controller source.

> **Research archive.** This is an anonymized research archive, not a portable installer or a claim about a currently running service. No GPU benchmarks were rerun while preparing this archive. The model/version labels and package versions below are those recorded in the original experiments.

## What the optimization actually changes

The final hook in [`scripts/sitecustomize-kda059.py`](scripts/sitecustomize-kda059.py) enables two additional shapes in the existing SGLang KDA GEMM eligibility table:

| Shape `(M, N, K)` | Operation |
| --- | --- |
| `(8, 17408, 5120)` | Down projection |
| `(8, 5120, 248320)` | LM head |

It uses existing upstream kernels, rather than claiming a new GEMM implementation. It checks the installed KDA source hash, loads verified FlashInfer AOT libraries and fails closed if the expected root or eligibility changes. The archive also contains unsuccessful experiments and alternatives; these are research records, **not recommendations to enable every patch**. In particular, the experimental positive-`min_p` path was not promoted to the final configuration.

## Recorded throughput

The final KDA2 candidate (`048`) completed **27/27 requests**, each with **1,024 output tokens**. Every throughput value below is the **median of three repetitions**; input counts are the API-reported counts, including prompt/template overhead.

| Task | Input tokens | Streaming estimate (tok/s) | End-to-end (tok/s) |
| --- | ---: | ---: | ---: |
| **Code** | **2,086** | **303.14** | **261.31** |
| Math | 2,085 | 232.24 | 225.84 |
| Prose | 2,086 | 167.67 | 163.20 |
| Code | 32,806 | 235.47 | 227.72 |
| Math | 32,805 | 232.05 | 223.67 |
| Prose | 32,806 | 143.96 | 140.11 |
| Code | 130,086 | 228.35 | 218.65 |
| Math | 130,086 | 181.30 | 174.01 |
| Prose | 130,086 | 127.01 | 123.93 |

**What these numbers mean:** `stream_tokens_per_second_estimate` divides API-reported completion tokens by the first-to-last nonempty content/reasoning chunk interval. It excludes initial waiting and can have first-chunk/hidden-token bias. The OpenAI stream did not expose per-chunk token counters, so the native `decode_tokens_per_second` field is **null**: **303.14 tok/s is a streaming estimate, not an instrumented GPU decode rate**. `e2e_tokens_per_second` includes the complete request duration, including initial waiting and the final usage-only delay. These medians cover three repetitions, not a cold-start-only benchmark; cache state was not exposed in the OpenAI usage records. The recorded generation settings were temperature `0`, top-p `1` and max output `1024`.

Source: [`performance048.jsonl`](evidence/performance048.jsonl), including its `metrics`, `metric_notes`, `usage` and `declared_settings` fields.

### Improvement against neighboring controls

The final candidate won all nine scenarios against controls measured immediately before and after it:

| Comparison | Geometric-mean streaming improvement | Scenarios improved |
| --- | --- | --- |
| Previous control | **+1.7421%** | 9/9 |
| Subsequent control | **+2.7542%** | 9/9 |

Evidence: [`pair047048-summary.json`](evidence/pair047048-summary.json), [`pair047048050-summary.json`](evidence/pair047048050-summary.json). These comparisons use per-scenario medians of the streaming estimate, not full-request end-to-end throughput or GPU-microbenchmark speedups.

The original final runbook records 27 performance cases with 1,024 output tokens, a 131,110-token context test, 3/3 exact needle responses, quality/tool-use checks and seeded sampling checks. Consult the actual evidence and protocol before extrapolating.

## Repository map

- [`OPTIMIZATION.md`](OPTIMIZATION.md): full experimental notebook, including hypotheses, failures and limitations.
- [`FINAL_RUNBOOK.md`](FINAL_RUNBOOK.md): historical final configuration and operational constraints.
- [`scripts/`](scripts/): hooks, GEMM microbenchmarks, end-to-end checks, profiling, experiment preparation and process-control scripts.
- [`evidence/`](evidence/): anonymized textual experiment captures and recorded results.
- [`windows/qwenctl/`](windows/qwenctl/): Go/Python controller source, installer and tests. Prebuilt binaries and nested Git history are deliberately not included.
- [`PUBLICATION_MANIFEST.json`](PUBLICATION_MANIFEST.json): export inventory, original/exported hashes, anonymization and exclusions.

## Reproduction boundary

The validated configuration recorded:

- RTX 5090; 460 W power limit;
- SGLang `0.5.19.dev992+g01e19ddf5`;
- FlashInfer `0.6.18`;
- Torch `2.13.0+cu130`;
- target `Qwen3.8-27B-NVFP4-LMHead4` and draft `DFlash2-NVFP4-RTNcal`;
- context length `131200`, token pool `131328`, DFlash2 block size `8`;
- the exact KDA source and nine AOT-library hashes recorded in the hooks/manifests.

**This archive does not contain model weights, AOT `.so` libraries, installed dependency snapshots or the original GPU machine.** Providing and validating those dependencies is required before reproducing the GPU result. Do not interpret a source checkout or passing controller tests as reproduction of the optimization.

Infrastructure identities and local user directories were anonymized. Archived scripts retain host-specific assumptions, pinned roots, process checks and operational commands; they are **not safe to execute blindly**. Review them, use resources you own exclusively and rerun all numerical, performance and quality gates for your environment. Do not point historical stop/recovery/preparation scripts at an existing or shared service. Follow the controller's documented operator flow instead of replaying old experiment launch scripts.

## Privacy and provenance

The original working directory was preserved. Only a separate export was anonymized. Benchmark numbers were preserved; anonymized files have different bytes, so their export hashes must not be confused with the original runtime hashes. Original captures describe historical machines and services, not live status.

Private PI sessions, Git metadata, caches, binaries, compressed raw archives, binary profiler captures and third-party source/install trees are excluded. Exclusions do not delete any original research material. Upstream kernels, frameworks and model artifacts are separate works and retain their own licenses; an MIT license for original research code must not be read as relicensing them.

See [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) for provenance and licensing boundaries. Original research code and documentation are available under the [MIT License](LICENSE). Upstream implementations, copied excerpts, tools and model artifacts retain their own licenses; see the notices before redistributing them.

## Export checks

The anonymized copy passed Go controller tests (8 workers, 4 s wall time) and all 10 Python controller tests (0.130 s, one worker), using isolated temporary fixtures and mocked operations. 288 Python source files, 127 JSON files and 811 JSONL records passed static parsing. The two final comparison summaries preserve their original numeric structure. No GPU benchmark, model inference, installation or live service operation was performed.
