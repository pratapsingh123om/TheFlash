.PHONY: test run

test:
	python -m compileall -q backend scripts tests
	python tests/dependency_manifest_test.py
	python tests/bootstrap_test.py
	python tests/smoke_test.py
	python tests/comparison_archive_test.py

run:
	python -m backend.serve
