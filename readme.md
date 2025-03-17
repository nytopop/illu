# illu - realtime conversational dynamics
4 models in a trenchcoat walk into a bar (3090 edition)

# usage
clone the repo and use `uv` or something

```shell
uv venv --python 3.12
uv pip install -r requirements.txt
uv run python -- illu.py
```

or this which basically does that:

```shell
make
```

have fun

# configuration
it checks `OPENAI_API_KEY` and `OPENAI_BASE_URL` env vars if you want to point it at whatever LLM. defaults to an
api on localhost:8000 (if unset) to make it convenient for me personally.

# resource reqs
~6.5 GB VRAM + whatever your LLM uses if you run that locally
