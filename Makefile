.PHONY: install run api test lint format clean

install:
	pip install -r requirements.txt
	python -m spacy download en_core_web_sm

run:
	python -m src.main --resume data/resumes/sample.pdf --jd data/job_descriptions/sample_jd.txt

api:
	uvicorn src.api:app --reload --port 8000

test:
	pytest tests/

lint:
	ruff check src/ tests/
	mypy src/

format:
	black src/ tests/
	ruff check --fix src/ tests/

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage
