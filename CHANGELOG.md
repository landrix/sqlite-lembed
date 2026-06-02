# Changelog

## 0.0.1-alpha.8-landrix.1

- Based on upstream `sqlite-lembed` `0.0.1-alpha.8`.
- Merged upstream PR #19: update `llama.cpp`, adapt the new llama.cpp API, and fix the build process.
- Merged upstream PR #21: prevent crashes for long inputs and return clearer embedding errors.
- Fixed Windows CMake builds by generating `sqlite-lembed.h` from `sqlite-lembed.h.tmpl`.
- Wrote Windows runtime DLLs to one output directory for easier Delphi deployment.
- Freed registered `llama_context` and `llama_model` instances before shutting down the llama backend.
