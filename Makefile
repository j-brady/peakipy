.PHONY: coverage

coverage:
	coverage run -m pytest test/test_core.py test/test_main.py test/test_fit.py test/test_cli.py

coverage-html:
	coverage html
	firefox htmlcov/index.html

test: coverage coverage-html
