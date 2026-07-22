# Migration Contract

Load this only for forensic migration work.

Process VEATIC 2.1, zero-label, AGAIN Phase 7 through Phase 0, original VEATIC, then relevant early exploration.

Before moving scientific material, record its source, destination, size, SHA-256 tree digest, role, and verification state in `move-manifest.csv`. Use an atomic rename on one volume; across volumes, verify the destination checksum before removing the source. Tree hashes exclude `.DS_Store`, `.pytest_cache`, and `__pycache__`.

Move coherent phase packages. Migrate only tests that protect the destination implementation, scientific contracts, reproducibility, or a demonstrated failure mode.
