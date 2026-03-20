PYTHON ?= python3
NPM ?= npm

VENV_DIR ?= .venv
FRONTEND_DIR := client
BACKEND_DIR := server
BACKEND_APP ?= main:app_sio
BACKEND_PORT ?= 8000

PIP := $(VENV_DIR)/bin/pip
PYTHON_BIN := $(VENV_DIR)/bin/python
UVICORN := $(VENV_DIR)/bin/uvicorn

.PHONY: help install install-client venv install-server compile-cython run-backend run-frontend start clean re

help:
	@echo "Available functions:"
	@echo "  make install       - Install frontend npm modules + backend Python deps in .venv"
	@echo "  make compile - Compile Cython files"
	@echo "  make run-backend   - Start backend"
	@echo "  make run-frontend  - Start frontend"
	@echo "  make start         - Start backend and frontend together"
	@echo "  make clean         - Remove Cython compiled files"
	@echo "  make re            - Clean and rebuild everything"

install: install-client compile

install-client:
	cd $(FRONTEND_DIR) && $(NPM) install

venv:
	@if [ ! -d "$(VENV_DIR)" ]; then \
		$(PYTHON) -m venv $(VENV_DIR); \
	fi

install-server: venv
	$(PIP) install --upgrade pip
	$(PIP) install -r $(BACKEND_DIR)/requirements.txt

compile: install-server
	cd $(BACKEND_DIR) && ../$(PYTHON_BIN) setup_ai_cython.py build_ext --inplace

run-backend: compile
	cd $(BACKEND_DIR) && ../$(UVICORN) $(BACKEND_APP) --reload --port $(BACKEND_PORT)

run-frontend: install-client
	cd $(FRONTEND_DIR) && $(NPM) run dev

start: install
	@backend_pid=; frontend_pid=; \
	trap ' \
		[ -n "$$backend_pid" ] && kill -TERM "$$backend_pid" 2>/dev/null || true; \
		[ -n "$$frontend_pid" ] && kill -TERM "$$frontend_pid" 2>/dev/null || true; \
		[ -n "$$backend_pid" ] && pkill -TERM -P "$$backend_pid" 2>/dev/null || true; \
		wait; \
	' INT TERM EXIT; \
	( cd $(BACKEND_DIR) && ../$(UVICORN) $(BACKEND_APP) --reload --port $(BACKEND_PORT) ) & backend_pid=$$!; \
	( cd $(FRONTEND_DIR) && $(NPM) run dev ) & frontend_pid=$$!; \
	wait

clean:
	rm -rf $(BACKEND_DIR)/build
	find $(BACKEND_DIR) -type f \( -name "*.so" -o -name "*.pyd" -o -name "*.dylib" \) -delete
	find $(BACKEND_DIR) -type d -name "__pycache__" -prune -exec rm -rf {} +
	find $(BACKEND_DIR)/ai -type f -name "*.c" -delete

re: clean install

# lsof -ti tcp:8000 / kill -9 pid