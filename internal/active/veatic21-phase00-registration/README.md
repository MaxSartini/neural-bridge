# VEATIC 2.1 Phase 00 Prospective Registration

Status: registered, not executed

This registration freezes the one-time construction of the 124-video consolidated Neural
Bridge input bundle and the model-free Phase 00 protected-input audit. Execution is forbidden
until the registration and implementation commit is present on `origin/main`.

The registered topology uses 12 CPU processes. It was selected by a full 124-video real-input
benchmark of validation, decompression, hashing, copying, and source re-verification. This
workload has no compatible GPU compute kernel; using the GPU would add transfers without
accelerating CSV parsing, ZIP decompression, SHA-256, or file I/O.

The execution request is [`experiment-registration.json`](experiment-registration.json).
