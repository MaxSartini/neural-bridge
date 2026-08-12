# Backend handoff

Written 12 August 2026, at the end of the frontend clean-up, for whoever picks up the backend
next. The aim throughout is **small changes with disproportionate payoff** — this codebase is
in good shape and does not want a rewrite.

---

## 1. Three things are called "the backend". Only one of them exists.

This matters before anything else, because the word points at three unrelated things and only
one is real code doing real work.

| | What it is | Size | Status |
|---|---|---|---|
| **Research engine** — `src/neural_bridge/` | The actual system. Trains and evaluates the bridge. | 3,686 lines Python | **Live.** The only real backend. |
| **Evidence server** — `dashboard/server/` | Read-only Express API serving Markdown/JSON/CSV to the internal dashboard. Four GET routes, `127.0.0.1` only. | ~400 lines TS | **Shelved** with the internal surface. |
| **Studio's "backend"** — `MockAnalysisClient` | Not a backend. An in-process fake behind an interface. No network, no server. | ~80 lines TS | **Does not exist yet.** |

The third row is the one that will bite. **The live, public, investor-facing product has no
backend at all.** Everything it shows is a fixture. That is deliberate and documented
(ADR-0003), the seam is now properly built, and the demo is honest about it — but "wire up the
backend" for the Studio is not a small task, it is the whole of Phase B/C, and nothing about it
has been started.

---

## 2. The research engine is well built. Say so before criticising it.

Genuinely good, and worth preserving through any future change:

**`again/contracts.py` is the right kind of module.** Frozen dataclasses, `Literal` unions, and
named constants — `BOUNDARY_POLICY = "use_annotation_covered_video_time_only"`,
`ROW_RATE_HZ = 2.0`, `ROW_STEP_SECONDS = 0.5`. Policy decisions that would otherwise be magic
numbers scattered across `data.py` and `replay.py` have one home and one name. This is the
codebase's shared vocabulary and it does the job `CONTEXT.md` does for the frontend.

**The training backend is a real seam, correctly placed.** `models.py` exposes
`Backend = Literal["auto","cpu","cuda","mlx"]`, `resolve_backend()`, and
`train_residual_head(...)`. The adapters — `torch_backend.py` (220 lines) and
`mlx_backend.py` (195 lines) — sit *behind* that interface, and the test exercises
`train_residual_head`, not the adapters. That is textbook: two adapters make the seam real
rather than hypothetical, and the interface is the test surface. Do not "improve" this by
testing the adapters directly.

**Module sizes are sane.** Fourteen of sixteen modules are under 420 lines, each with a clear
job: `data`, `engine`, `replay`, `models`, `evidence`, `checkpoints`, `metrics`, `configs`.

**Provenance is a first-class concern** — a dedicated `provenance.py`, and `mlflow_registry.py`
for run tracking. For a research codebase whose entire value is defensible claims, that is the
right thing to have invested in.

---

## 3. The headline finding: the backend seam's test runs zero cases

`tests/again/test_training_backends.py` is the only test of the training backends. It
parametrises two cases, and **both skip on this machine and almost certainly in CI**:

```
SKIPPED [1] test_training_backends.py:18: training extra not installed
SKIPPED [1] test_training_backends.py:18: MLX is available only with the training extra on macOS
```

The cause is structural, not an oversight in the test:

- `torch` and `mlx` live in `[project.optional-dependencies] training`, not in the dev group.
- CI runs `uv sync --dev`, which does not install optional extras.
- So `importlib.util.find_spec("torch") is None` → the `cpu` case skips.
- `platform.system() != "Darwin"` on `ubuntu-latest` → the `mlx` case skips.

**Net effect: the pipeline that gates every commit exercises the training path zero times.**
The suite reports green while testing none of it.

Two further gaps in the same seam:

- **`cuda` is in the `Backend` union and is never tested at all** — no parametrisation, no skip,
  no mention.
- **`resolve_backend("auto")` — the default every real caller uses — is never exercised.** The
  test passes explicit `"cpu"` and `"mlx"`, so the resolution logic that decides what actually
  runs is untested even when the extra *is* installed.

This is the same inversion the frontend review kept finding: the pure thing is tested, the
thing that decides what happens is not.

### The small, qualitative fix

Do not install torch in CI — that would add minutes to every run for little gain. Instead:

1. **Test `resolve_backend` directly.** It is pure: a requested value plus environment
   availability in, a concrete backend out. It needs no torch, no GPU and no macOS, and it is
   the function whose wrongness would silently route training to the wrong device. Perhaps
   fifteen lines of test for the highest-value uncovered logic in the repo.
2. **Make the skip loud.** A skipped test that nobody notices is indistinguishable from a
   passing one. Either add a CI step that asserts the training tests actually ran on the
   platform where they can, or emit a warning when the whole backend suite skips.
3. **Decide about `cuda` explicitly.** If nothing runs on CUDA, remove it from the `Literal` —
   an untested branch in a type union is a promise the code does not keep. If it is used,
   it needs at least the same skip-marked smoke test the others have.

---

## 4. `mlflow_registry.py` is 18% of the backend in one file

671 lines, 34 top-level functions and methods — roughly twenty lines each, so it is not a
god-object, but it is by a wide margin the largest module and it sits at the root of the
package rather than inside a subpackage like `again/` or `zero_label/`.

Two observations rather than a recommendation, because splitting it without a reason is churn:

- **It is the only root-level module doing substantial work.** `provenance.py` (75 lines) is
  the other root module, and the two are plausibly the same concern — recording what ran and
  proving it. If either is touched substantially, consider a `tracking/` subpackage.
- **It owns the one genuinely failing test.** See below.

**Do not split it speculatively.** It has tests, it has a clear job, and 671 lines is large but
not unmanageable. Split it when a change makes the seam obvious, not before.

---

## 5. One real test failure, and it is environmental

```
FAILED tests/test_mlflow_registry.py::test_tracking_uri_uses_absolute_sqlite_path
AssertionError: assert 'sqlite:///C:...t0\\mlflow.db' == 'sqlite:///C:...ut0/mlflow.db'
```

The test builds an expected URI with an f-string containing a literal `/`; on Windows
`Path` yields `\`. **CI passes** — it runs `ubuntu-latest`. But it fails for anyone developing
on Windows, which is where this repo currently lives, and a suite that is red on your own
machine is a suite you stop reading.

**Fixed in this handoff.** The test now builds its expectation the way the implementation does
instead of gluing a literal `/` onto a `Path`, and additionally asserts the property the test
name actually claims — that the embedded path is absolute. Local suite is green: 38 passed,
2 skipped.

**One question deliberately left open.** `tracking_uri` produces `sqlite:///C:\Users\...` on
Windows, with backslashes inside a URI. SQLAlchemy tolerates this, and no training has ever
run on Windows here, so changing the implementation would be an unvalidated behaviour change
to fix a problem nobody has hit. It is recorded rather than acted on. If Windows ever becomes
a real training platform, normalising to forward slashes is the change — and it needs testing
against MLflow, not just against the unit test.

---

## 6. What the Studio will actually need, when you get there

Not tomorrow's job, but the shape is already decided and worth writing down while it is fresh.

`studio/src/api/analysisClient.ts` is the contract. A real backend has to satisfy it, and the
interface deliberately encodes three honesty rules that the mock already respects:

- **Reject with `AnalysisNotFound` for an unknown id.** Do not return a plausible empty report.
- **Omit `PipelineStep.progress` when a step cannot measure itself.** The UI draws no bar rather
  than a fake percentage. A backend that reports a made-up 50% breaks a promise the interface
  makes.
- **Omit `etaSeconds` when there is no honest estimate.** The screen drops the line rather than
  guessing.

Two constraints that will shape the architecture more than anything else:

- **ADR-0001 forbids putting inference in `dashboard/server/`.** That server is read-only by
  contract, four GET routes, localhost-bound. A real inference API is a separate service. Do
  not reopen this casually; the read-only guarantee is what makes the evidence server safe.
- **The Studio currently ships with `connect-src 'none'`.** Adding a real backend means
  relaxing the CSP to allow exactly one origin, and that is a deliberate security change that
  wants an ADR — the current policy is what lets a fully public, unauthenticated demo be safe.

Also worth flagging: `AnalysisInput` collects `contentType`, `objective` and `notes`, and the
mock discards all three. A real backend is the first thing that would use them. Either wire
them up or drop them from the interface — collecting input that goes nowhere is a small lie in
the UI.

---

## 7. Recommended order, smallest first

| # | Change | Effort | Why it is worth it |
|---|---|---|---|
| 1 | Fix the Windows path assertion in `test_mlflow_registry.py` | ~10 min | Restores a green local suite. A red suite you learn to ignore is worse than no suite. |
| 2 | Test `resolve_backend()` directly | ~30 min | Highest-value uncovered logic in the repo. Pure, no extras needed, and it decides what hardware actually runs. |
| 3 | Resolve `cuda` — test it or remove it from the union | ~15 min | An untested branch in a type union is a promise the code does not keep. |
| 4 | Make an all-skipped backend suite visible | ~30 min | Today the suite is green while testing none of the training path. That is the failure mode worth closing. |
| 5 | Decide on `AnalysisInput`'s three unused fields | ~15 min | Either wire them or drop them; do not keep collecting input that goes nowhere. |

Everything above is under two hours combined and none of it is a refactor.

## 8. What not to do

- **Do not split `mlflow_registry.py` for its size alone.** Wait for a change that reveals the
  seam.
- **Do not test `torch_backend.py` and `mlx_backend.py` directly.** They are adapters behind a
  correctly-placed interface; testing through `train_residual_head` is right, and reaching past
  it would make the seam harder to move.
- **Do not install the training extra in CI** to make the skips go away. It buys a slow smoke
  test; testing `resolve_backend` buys the actual logic.
- **Do not add write endpoints to `dashboard/server/`.** ADR-0001, and it is load-bearing.
- **Do not touch `src/neural_bridge/veatic21` or restart VEATIC 2.1 work** — the root
  `AGENTS.md` and `CURRENT_STATE.md` govern that, and `tests/test_veatic21_authority.py`
  enforces it.
