# Changelog

## 0.0.1-alpha.8-landrix.1

Fork of upstream [`asg017/sqlite-lembed`](https://github.com/asg017/sqlite-lembed),
based on `0.0.1-alpha.8` (commit `4e3589a`), adapted for use in
[`sqlite-vec-for-Delphi`](https://github.com/landrix/sqlite-vec-for-Delphi).

### Updated `llama.cpp` (upstream PR [#19](https://github.com/asg017/sqlite-lembed/pull/19))

- Bumped the `vendor/llama.cpp` submodule (`2b33896` → `4fd1242`).
- Adapted `sqlite-lembed.c` to the new `llama.cpp` API:
  - `llama_tokenize(model, …)` now takes a vocab handle via `llama_model_get_vocab(model)`.
  - `llama_n_embd` → `llama_model_n_embd`.
  - `llama_kv_cache_clear(context)` → `llama_memory_clear(llama_get_memory(context), false)`.
  - `llama_token_get_score` → `llama_vocab_get_score`.
  - `llama_token_to_piece` updated to the new signature (added `lstrip` argument).
  - `llama_load_model_from_file` → `llama_model_load_from_file`,
    `llama_new_context_with_model` → `llama_init_from_model`,
    `llama_free_model` → `llama_model_free`.

### Crash fix for long inputs and clearer errors (upstream PR [#21](https://github.com/asg017/sqlite-lembed/pull/21))

- `embed_single()` now rejects inputs whose token count exceeds `n_ctx` instead of
  overflowing a fixed-size batch (previously a hardcoded `n_batch = 512`), preventing a
  segfault on long inputs.
- The batch is sized dynamically via `llama_batch_init(token_count, 0, 1)`.
- Added an `out_error` parameter so failures (tokenization, decode, embedding extraction,
  input too long) surface as descriptive messages through `sqlite3_result_error()` instead
  of a generic "Error generating embedding".

### Memory cleanup on shutdown

- `api_free()` now frees each registered `llama_context` (`llama_free`),
  `llama_model` (`llama_model_free`) and the model `name` **before** calling
  `llama_backend_free()`, avoiding use-after-free / leaks on extension unload.

### Build improvements

- `CMakeLists.txt` generates `sqlite-lembed.h` from `sqlite-lembed.h.tmpl` into a
  `generated/` include dir (fixes Windows builds where the header was not produced).
- Runtime artifacts are written to a single `bin/` (and `lib/`) output directory for
  easier Delphi deployment.
- The shared library links against `ggml` instead of `ggml_static`.
- `Makefile` passes `-DCMAKE_OSX_ARCHITECTURES` for x86_64 and arm64 macOS targets
  (arm64 support).

### Tests and tooling

- `tests/test-loadable.py` implements previously skipped pytest cases (tokenization,
  `token_score`, `token_to_piece`, `chunks`, `model_size`, `models`) and was made robust
  across platforms.
- `.gitignore` ignores `build/`, `build-*/` and `tests/__pycache__/`.
