# Neural Bridge Project Memory

## 2026-06-13 13:35:34 - VEATIC 50 Run And Automation Handoff

- Handoff file: `docs/handoffs/20260613_133534_veatic-50-run-and-automation-handoff.md`
- Notes: Added durable handoff/memory generator and VEATIC gated pipeline wrapper. Current VEATIC 50 cortical-only extraction is running/resumable; subcortical remains disabled by default.
- VEATIC cache: complete=26, failed=0, incomplete=[{'video_id': '100', 'duration_seconds': 53.28, 'error': None}]
- Standing guardrail: real cortical must beat autoregressive, shuffled cortical, and random Gaussian controls before scale-up claims.

## 2026-06-15 12:32:25 - VEATIC 50 Complete And Cached Feature Gates

- Handoff file: `docs/handoffs/20260615_123225_veatic-50-complete-and-feature-gates.md`
- VEATIC first-50 cortical cache is complete: 50/50 valid raw `tribe_raw_output.npz` files and 50/50 complete status files under `/Volumes/onn. Drive/Neural Bridge/benchmarks/veatic/tribe_cache`.
- The final missing videos were encoded one at a time with MLX V-JEPA2: `70`, `10`, and `7`.
- Cached 50-video default gate completed: `accepted_videos=50`, `accepted_rows=2489`, gate decision `scale_candidate`.
- Default 6-feature `cortical_global` gate passed 11/51 checks, 10 robust non-official checks. Strongest evidence is blocked temporal-gap arousal dynamics; official 70/30 future-change checks did not pass.
- Richer no-reencode feature modes also completed from cache: `cortical_global_delta`, `cortical_pca_64`, and `cortical_pca64_delta`.
- Best blocked temporal-gap arousal future-change result among these was `cortical_pca64_delta`: p1 MAE `0.0410` vs autoregressive `0.0664`; p2 MAE `0.0675` vs `0.1316`; p3 MAE `0.0913` vs `0.1705`, also beating shuffled and random controls.
- Current honest verdict: promising scale candidate / partial pass, not final proof. Next step is a stability report across feature modes and tougher validation before making investor-grade claims.
