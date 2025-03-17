.PHONY: run deps clean

run: .venv
	uv run python -- illu.py

.venv:
	uv venv --python 3.12
	uv pip install -r requirements.txt

deps: .venv
	uv pip install -r requirements.txt

clean:
	rm -rf .venv __pycache__
