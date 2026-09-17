install:
	pip install --upgrade pip && pip install -r requirements.txt

train:
	python src/train.py

test:
	pytest tests/

run-api:
	uvicorn api.app:app --reload --host 0.0.0.0 --port 8000
