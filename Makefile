.PHONY: help install venv test run start docker-build docker-up docker-down clean

HOST ?= 127.0.0.1
PORT ?= 8000

ifeq ($(OS),Windows_NT)
VENV_PY := .venv/Scripts/python.exe
else
VENV_PY := .venv/bin/python
endif
PYTHON := $(if $(wildcard $(VENV_PY)),$(VENV_PY),python)

help:
	@echo "Targets:"
	@echo "  install       Install dependencies from requirements.txt"
	@echo "  venv          Create local .venv"
	@echo "  test          Run mocked test suite (no model download)"
	@echo "  run           Start API with uvicorn (HOST=$(HOST) PORT=$(PORT))"
	@echo "  start         Alias for run"
	@echo "  docker-build  Build container image"
	@echo "  docker-up     Start stack (build if needed)"
	@echo "  docker-down   Stop stack"
	@echo "  clean         Remove caches and bytecode"

install:
	$(PYTHON) -m pip install -r requirements.txt

venv:
	python -m venv .venv

test:
	$(PYTHON) -m pytest -q

run:
	$(PYTHON) -m uvicorn app.main:app --host $(HOST) --port $(PORT)

start: run

docker-build:
	docker compose build

docker-up:
	docker compose up --build

docker-down:
	docker compose down

clean:
	$(PYTHON) -c "import pathlib, shutil; [shutil.rmtree(p, ignore_errors=True) for p in list(pathlib.Path('.').rglob('__pycache__')) + [pathlib.Path('.pytest_cache')]]"
