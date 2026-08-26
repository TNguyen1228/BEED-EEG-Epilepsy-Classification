.PHONY: help install train test serve lint docker-build docker-up

help:
	@echo "Available commands:"
	@echo "  make install      - Install Python dependencies"
	@echo "  make train        - Train Random Forest pipeline and log to MLflow"
	@echo "  make test         - Run pytest test suite"
	@echo "  make serve        - Start FastAPI development server"
	@echo "  make mlflow       - Start local MLflow UI"
	@echo "  make lint         - Check code style"
	@echo "  make docker-build - Build Docker image"
	@echo "  make docker-up    - Run API + MLflow UI with Docker Compose"

install:
	pip install -r requirements.txt

train:
	python -m src.models.train

test:
	pytest tests/ -v

serve:
	uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload

mlflow:
	mlflow ui --port 5000

lint:
	flake8 src api tests
	black --check src api tests

docker-build:
	docker build -t beed-eeg-epilepsy-api:latest .

docker-up:
	docker compose up -d

