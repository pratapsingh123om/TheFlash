.PHONY: test run

test:
	python -m compileall -q backend tests
	python tests/bootstrap_test.py
	python tests/smoke_test.py

run:
	python -m backend.serve
