VENV_NAME := venv
REQUIREMENTS := requirements.txt

activate:
	@echo "Setting up virtual environment..."
	 python3 -m venv $(VENV_NAME) && \
	 . $(VENV_NAME)/bin/activate && \
	 pip install --upgrade pip && \
	 pip install -r $(REQUIREMENTS)
	@echo "\nVirtual environment ready. To activate manually:"
	@echo "source $(VENV_NAME)/bin/activate"

fetch_tools:
	python3 tool_downloader.py


.PHONY: activate fetch_tools
