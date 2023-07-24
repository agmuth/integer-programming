format:
	poetry run black intprog/
	poetry run isort intprog/
	
	poetry run black tests/
	poetry run isort tests/
	
lint: 
	poetry run ruff check intprog/. --fix
	poetry run ruff check tests/. --fix

test:
	poetry run pytest tests/.