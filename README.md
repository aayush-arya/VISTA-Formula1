# VISTA

**Video-Informed Strategy and Telemetry Assistant**
A video-grounded race strategy copilot for Formula 1.

VISTA turns real race footage into a model of what is happening on track, aligns it with real
telemetry, predicts strategy-relevant quantities, and uses an LLM agent with rule retrieval to
explain and recommend decisions with traceable numbers and citations.

## Modules
| Module | Path | Status |
|---|---|---|
| Data pipeline (FastF1) | `src/vista/data` | started |
| Tyre degradation baseline | `src/vista/models` | started |
| Perception (YOLOv8 + ByteTrack, OCR) | `src/vista/perception` | planned |
| Video-telemetry sync | `src/vista/sync` | planned |
| Copilot (tools + RAG) | `src/vista/copilot` | planned |
| Evaluation and ablations | `src/vista/evaluation` | planned |
| Dashboard and overlay renderer | `src/vista/app` | planned |

## Quickstart
```bash
make setup            # virtualenv + dependencies
source .venv/bin/activate
make data             # download laps for races in configs/races.yaml
make baseline         # leave-one-race-out tyre baseline
make test
```

## Conventions
- Split by race, never by lap. One race is held out for final reporting.
- Every experiment has a config in `configs/`, a fixed seed and an entry in `docs/EXPERIMENT_LOG.md`.
- Raw footage and caches are never committed (see `.gitignore` and `docs/DATA_STATEMENT.md`).
- Each trained model gets a short model card in `docs/`.

## Phase 1 checklist (weeks 1 to 3)
- [ ] Run `make data` for 3 candidate races and inspect safety cars, weather and stints
- [ ] Freeze the race list and choose the held-out final test race
- [ ] EDA notebook: stint lengths, degradation per compound, pit windows
- [ ] Run `make baseline` and record results in the experiment log
- [ ] Confirm footage licensing with the supervisor
- [ ] Fill in `docs/DATA_STATEMENT.md`
