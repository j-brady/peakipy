.PHONY: test

test:
	pytest test/test_core.py
	pytest test/test_main.py
	pytest test/test_fit.py
	pytest test/test_cli.py
