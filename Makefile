.PHONY: coverage

coverage:
	coverage run -m pytest test/test_fitting.py \
                   test/test_lineshapes.py \
  		   test/test_io.py \
		   test/test_utils.py \
		   test/test_main.py \
		   test/test_cli.py \
		   test/test_simulation.py \
		   test/test_plotting.py

coverage-html:
	coverage html
	firefox htmlcov/index.html

test: coverage coverage-html
