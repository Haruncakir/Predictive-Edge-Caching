# ML Predictor & Edge Cache Controller

Python module for the **Predictive Edge Caching** pipeline (CSE 476/575).

Implements the "ML Predictor" and "Edge Cache Controller" blocks from the
architecture diagram.

## Setup (requires [uv](https://docs.astral.sh/uv/))

```bash
cd ml/
uv sync                # creates .venv and installs all deps
uv sync --extra dev    # also installs pytest
```

## Quick Start

```bash
# 1. Generate a trace from the C++ simulator (from the repo root):
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build
./build/request_sim export 50000

# 2. Run the ML vs baseline comparison:
cd ml/
uv run pec-simulate --trace ../data/generated_trace.csv --cache-mb 50

# 3. Generate plots:
uv run python scripts/plot_results.py ../results/
```

## Architecture

```
                      ┌─────────────────────────────────┐
  trace.csv ──────▶   │        Simulator (simulator.py)  │
                      │                                   │
                      │  ┌────────────┐  ┌─────────────┐ │
                      │  │  Feature    │  │  ML Model   │ │
                      │  │  Extractor  │──│  (XGBoost)  │ │
                      │  └────────────┘  └──────┬──────┘ │
                      │                          │        │
                      │  ┌────────────────────────▼─────┐ │
                      │  │  Edge Cache Controller       │ │
                      │  │  ┌────────┐  ┌────────────┐  │ │
                      │  │  │ Store  │  │ Policy     │  │ │
                      │  │  │ (byte) │  │ (LRU/ML/…) │  │ │
                      │  │  └────────┘  └────────────┘  │ │
                      │  └──────────────────────────────┘ │
                      │                                   │
                      │  ┌──────────────────────────────┐ │
                      │  │  Metrics Reporter → CSV      │ │
                      │  └──────────────────────────────┘ │
                      └───────────────────────────────────┘
                                      │
                                ▼  results/*.csv
                        plot_results.py → PNG/PDF graphs
```

## Running Tests

```bash
cd ml/
uv run pytest tests/ -v
```

## Module Structure

| File | Description |
|------|-------------|
| `src/pec/types.py` | Request, Feedback, FileFeatures dataclasses |
| `src/pec/feature_extractor.py` | 9-feature per-file vector extraction |
| `src/pec/model.py` | XGBoost predictor with online training |
| `src/pec/cache/store.py` | Byte-addressed fixed-capacity store |
| `src/pec/cache/policies.py` | LRU, LFU, FIFO, ML-Driven policies |
| `src/pec/cache/controller.py` | Orchestrates store + policy + ML |
| `src/pec/metrics.py` | Time-series metrics logger |
| `src/pec/simulator.py` | Main simulation loop |
| `src/pec/cli.py` | CLI entry point |
| `scripts/plot_results.py` | Matplotlib comparison graphs |
