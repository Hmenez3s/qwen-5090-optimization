# Provenance and third-party boundaries

This archive contains research instrumentation, process-local hooks and historical experiment records. It does **not** claim authorship of upstream GPU kernels. The MIT license in `LICENSE` applies to original research code and documentation only. It does not replace the licenses of imported implementations, model artifacts, copied excerpts or third-party tools. An Apache-2.0 license copy is provided in [`licenses/Apache-2.0.txt`](licenses/Apache-2.0.txt) for the historical SGLang/FlashInfer source excerpts identified here.

## Recorded upstream references

### NInfer

The original research captured NInfer commit:

`b88c0f6fc7e999f13eb2fcf7fc9105ed79a91868`

- Immutable reference: https://github.com/Neroued/ninfer/commit/b88c0f6fc7e999f13eb2fcf7fc9105ed79a91868
- Repository: https://github.com/Neroued/ninfer
- The LICENSE in the captured source tree states **Apache License 2.0**.
- That tree also contains dependencies with their own MIT/Unicode notices; it is not a single-license source bundle.

The captured third-party source tree, its compressed archive and upstream README copies are **not redistributed here**. This reference records the provenance observed in the research; current remote availability and a remote byte-for-byte comparison were not checked during export.

### SGLang / FlashInfer

Recorded source retrieval used SGLang revision `01e19ddf5` for sampling sources and associated metadata. The sampled source headers state Apache-2.0 and credit SGLang/FlashInfer contributors. These observations are not a blanket license audit of all installed KDA or Blackwell kernel versions.

- SGLang: https://github.com/sgl-project/sglang
- FlashInfer: https://github.com/flashinfer-ai/flashinfer

The final optimization changes **eligibility and launch/admission configuration** around existing implementations. The `micro_gemm*.py` files measure and wrap imported backends; they are not standalone CUDA/CuTe GEMM implementations. Imported kernels and AOT libraries are not redistributed.

### Other runtime prerequisites

PyTorch, CUTLASS/CuTe DSL, CUDA, NVIDIA drivers, model weights and draft-model artifacts are separate prerequisites. Obtain them from their own publishers and comply with the licenses of the exact versions/artifacts you use. Their licenses and redistribution rights must not be inferred from this repository's license.

## Excluded material

The export excludes vendored/installed dependency trees, `evidence/source-004.json` (an installed-source snapshot), binary profiles, raw compressed bundles, model weights, AOT shared libraries, controller binaries, private session state and nested Git metadata.

Short source excerpts that appear inside historical records retain their original provenance and upstream licensing. When reusing or redistributing an upstream excerpt, obtain and preserve its applicable license, copyright/NOTICE information and modification notices. Do not describe the archive as a newly authored kernel library or as a relicensing of upstream software.

## Scope of verification

Local source content and historical evidence were inspected. Controller tests can validate the anonymized source against mocks and temporary fixtures; they cannot validate GPU behavior, model quality, original operational receipts or portability of the historical machine-specific setup. Exact-version licensing for all installed KDA/Blackwell dependencies and remote availability remain unverified.
