.PHONY: help install install-dev security-scan clean

help:
	@echo "Available commands:"
	@echo "  make install        - Install production dependencies"
	@echo "  make install-dev    - Install development dependencies"
	@echo "  make security-scan  - Run security vulnerability scan"
	@echo "  make clean          - Remove cache and temporary files"

install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements.txt
	pip install -r requirements-dev.txt

security-scan:
	@echo "Running pip-audit..."
	pip-audit -r requirements.txt --desc
	@echo ""
	@echo "Security scan complete!"

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	@echo "Cleaned up cache and temporary files"
