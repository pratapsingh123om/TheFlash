from __future__ import annotations

import tomllib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
pyproject_dependencies = {
    dependency.strip() for dependency in project["project"]["dependencies"]
}
requirements_dependencies = {
    line.strip()
    for line in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
    if line.strip() and not line.lstrip().startswith("#")
}

if pyproject_dependencies != requirements_dependencies:
    raise SystemExit(
        "Dependency manifests differ: "
        f"pyproject_only={sorted(pyproject_dependencies - requirements_dependencies)}, "
        f"requirements_only={sorted(requirements_dependencies - pyproject_dependencies)}"
    )

print("DEPENDENCY MANIFEST TEST PASSED")
