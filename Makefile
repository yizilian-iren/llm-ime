PY := .venv/bin/python
ENGINE ?= ngram
ARGS ?=

.PHONY: eval-data eval repl

# Rebuild data/private/eval.jsonl (user_selected.jsonl + corrections are fixed by index; don't re-extract).
eval-data:
	$(PY) scripts/build_eval.py

# make eval ENGINE=llm ARGS="--limit 100"
eval:
	$(PY) scripts/run_eval.py --engine $(ENGINE) $(ARGS)

repl:
	$(PY) scripts/repl.py $(ARGS)
