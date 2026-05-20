uv run python scripts/generate_regime_trace.py -n 50000 --catalog-size 500 --hot-pool-size 50 --hot-fraction 0.80 --num-epochs 10

uv run pec-simulate --trace ../data/regime_trace.csv --cache-mb 25 --history-window 512
