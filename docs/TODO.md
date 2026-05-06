# TODO — HW1: Signal Frequency Extraction using Neural Networks

| Field    | Value                                                            |
|----------|------------------------------------------------------------------|
| Version  | 1.10                                                             |
| Updated  | 2026-04-28                                                       |
| Notes    | Decomposed into ≥ 75 atomic sub-tasks; max source-file length: **145 code lines** (was 150 in v1.00) |

Legend: 🔴 High | 🟡 Medium | 🟢 Low | ✅ Done | 🔄 In Progress | ⬜ Not Started

**Cross-reference notation**
- `PRD §X`     — section X of `docs/PRD.md`
- `PLAN §X`    — section X of `docs/PLAN.md`
- `PRD_sig`    — `docs/PRD_signal_generation.md`
- `PRD_mod`    — `docs/PRD_models.md`
- `PRD_tr`     — `docs/PRD_training.md`

**Constraint update (v1.10):** maximum Python source-file length reduced from
150 → **145 code lines** (excluding blank lines and comment-only lines). This is
enforced by NFR-3 and audit task 8.7. Tests files obey the same limit.

---

## Phase 1 — Documentation

| #    | Task                                          | Pri | Status | Refs               | Definition of Done                                              |
|------|-----------------------------------------------|-----|--------|--------------------|-----------------------------------------------------------------|
| 1.1  | Create docs/PRD.md with all 15 sections       | 🔴  | ✅     | PRD all            | Sections 1–15 present; KPI table; FR-1..FR-8; NFR-1..NFR-7      |
| 1.2  | Update PRD NFR-3 + KPI to 145-line limit       | 🔴  | ✅     | PRD §5, §8         | NFR-3 reads "≤ 145 code lines"; KPI row updated                 |
| 1.3  | Create docs/PLAN.md with C4 + ADRs            | 🔴  | ✅     | PLAN all           | Context, Container, Component diagrams; 5 ADRs                  |
| 1.4  | Create docs/PRD_signal_generation.md          | 🔴  | ✅     | PRD FR-1, FR-2     | Inputs, outputs, 7 SG-T* test scenarios documented              |
| 1.5  | Create docs/PRD_models.md                     | 🔴  | ✅     | PRD FR-3..FR-5     | All 3 architectures with parameter counts and rationale         |
| 1.6  | Create docs/PRD_training.md                   | 🔴  | ✅     | PRD FR-6           | Loss, optimizer, schedule, early stop, 7 TR-T* scenarios        |
| 1.7  | Create docs/PROMPTS.md                        | 🟡  | ✅     | All                | All AI prompts logged with purpose, output, lessons             |
| 1.8  | Decompose docs/TODO.md to ≥ 300 lines         | 🔴  | ✅     | All                | Atomic sub-tasks; cross-refs to PRD/PLAN                        |
| 1.9  | Document gatekeeper extension points          | 🟡  | ✅     | PLAN §11           | Extension matrix updated with HTTP-adapter scaffold notes       |
| 1.10 | Add Bibliography section to PRD               | 🟢  | ✅     | PRD §15            | Cite Hochreiter 1997, Adam paper, Nyquist                       |

---

## Phase 2 — Project Scaffolding

| #    | Task                                          | Pri | Status | Refs               | Definition of Done                                              |
|------|-----------------------------------------------|-----|--------|--------------------|-----------------------------------------------------------------|
| 2.1  | Create pyproject.toml with project metadata   | 🔴  | ✅     | PRD §11            | name, version, deps; `uv sync` resolves cleanly                 |
| 2.2  | Configure ruff in pyproject.toml              | 🔴  | ✅     | PRD KPI            | line-length=100, target=py310, rules E,F,W,I,N,UP,B,C4,SIM      |
| 2.3  | Configure pytest + coverage                   | 🔴  | ✅     | NFR-5              | fail_under=85; source=src; omit main.py                         |
| 2.4  | Create config/setup.json (v1.00)              | 🔴  | ✅     | PRD FR-1..FR-7     | All hyperparameters present                                     |
| 2.5  | Create config/rate_limits.json (v1.00)        | 🔴  | ✅     | PLAN §6            | default, file_io, checkpoint services defined                   |
| 2.6  | Create config/logging_config.json (v1.00)     | 🔴  | ✅     | PLAN §8            | Console + file handlers; standard + detailed formatters         |
| 2.7  | Create .env-example                           | 🔴  | ✅     | NFR-4              | Placeholders only; commented                                    |
| 2.8  | Create .gitignore                             | 🔴  | ✅     | NFR-4              | .env, *.key, *.pem, secrets.*, __pycache__, .venv/              |
| 2.9  | Create directory tree                         | 🔴  | ✅     | PLAN §1            | src/, tests/, docs/, data/, results/, assets/, notebooks/       |
| 2.10 | Add LICENSE (MIT)                             | 🟡  | ✅     | PRD §15            | MIT text with author + year                                     |
| 2.11 | Add CITATION.cff (optional)                   | 🟢  | ✅     | PRD §15            | Machine-readable citation                                       |

---

## Phase 3 — Shared Infrastructure

| #    | Task                                          | Pri | Status | Refs               | Definition of Done                                              |
|------|-----------------------------------------------|-----|--------|--------------------|-----------------------------------------------------------------|
| 3.1  | Create shared/version.py with CODE_VERSION    | 🔴  | ✅     | PLAN §6            | CODE_VERSION="1.00"; parse_version(); validate_config_version() |
| 3.2  | Add MIN/MAX config-version constants          | 🔴  | ✅     | PLAN §6            | Out-of-range raises ValueError with message                     |
| 3.3  | Create shared/config.py with cached loader    | 🔴  | ✅     | PLAN §6            | get_setup(), get_rate_limits(), get_logging_config()            |
| 3.4  | Honour env-var overrides for config dir       | 🔴  | ✅     | NFR-4              | FREQ_EXTRACTOR_CONFIG_DIR respected                             |
| 3.5  | Add setup_logging() helper                    | 🔴  | ✅     | PLAN §8            | dictConfig from logging_config.json; ensures log dir exists     |
| 3.6  | Create shared/gatekeeper.py with ApiGatekeeper| 🔴  | ✅     | PLAN §6, ADR-4     | execute(), get_queue_status(), exponential-backoff retries      |
| 3.7  | Implement sliding-window rate limiter         | 🔴  | ✅     | PLAN §6            | 60-second window; sleeps when limit reached                     |
| 3.8  | Add singleton get_gatekeeper() factory        | 🔴  | ✅     | PLAN §6            | One instance per service name; cached                           |
| 3.9  | Create constants.py                           | 🔴  | ✅     | PLAN §6            | MODEL_TYPES, FREQUENCY_LABELS, SPLIT_NAMES, TENSOR_DTYPES       |
| 3.10 | Create __init__.py files in every package     | 🔴  | ✅     | NFR-6              | freq_extractor, sdk, services, shared all importable            |
| 3.11 | Define __all__ in package __init__.py         | 🟡  | ✅     | NFR-6              | Explicit public API surface                                     |
| 3.12 | Add module-level docstrings to all shared/*   | 🔴  | ✅     | NFR-3              | Every shared module has a docstring                             |
| 3.13 | Verify each shared file ≤ 145 code lines      | 🔴  | ✅     | NFR-3 (v1.10)      | Line counter confirms compliance                                |
