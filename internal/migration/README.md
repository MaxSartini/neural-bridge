# Local-First Migration

The existing GitHub repository is an incomplete input. The migration audits all locally available internal and external material, then moves verified files into this repository or the external artifact root.

Order:

1. VEATIC 2.1, from today's locked state backward.
2. Zero-label confirmation and discovery.
3. AGAIN, Phase 7 backward through Phase 0.
4. Original VEATIC.
5. Earliest relevant exploration and infrastructure.

No scientific file moves during discovery. Every move must be declared in the ledger, verified after relocation, and completed as a coherent phase package. The root README is written last from settled packages.

Tree hashes exclude only `.DS_Store`, `.pytest_cache`, and `__pycache__`; these are filesystem/runtime residue, never evidence.

Tests are selected after the destination implementation is known. Do not copy a source test directory wholesale; retain only the smallest suite that proves the migrated endpoint and its scientific contracts.
