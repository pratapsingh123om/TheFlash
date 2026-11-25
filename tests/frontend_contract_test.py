from __future__ import annotations

import re
from collections import Counter
from html.parser import HTMLParser
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class IdCollector(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.ids: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        if attributes.get("id"):
            self.ids.append(str(attributes["id"]))


def main() -> None:
    html = (ROOT / "frontend" / "index.html").read_text(encoding="utf-8")
    javascript = (ROOT / "frontend" / "app.js").read_text(encoding="utf-8")

    collector = IdCollector()
    collector.feed(html)
    duplicates = sorted(
        identifier
        for identifier, count in Counter(collector.ids).items()
        if count > 1
    )
    if duplicates:
        raise AssertionError(f"Duplicate frontend IDs: {duplicates}")

    referenced_ids = set(
        re.findall(r'getElementById\("([^"]+)"\)', javascript)
    )
    missing = sorted(referenced_ids - set(collector.ids))
    if missing:
        raise AssertionError(f"JavaScript references missing HTML IDs: {missing}")

    required_copy = (
        "No live feed configured",
        "Corrected 2023 comparison archive is pending",
        "2023 Development Evidence",
    )
    combined = html + javascript
    absent_copy = [text for text in required_copy if text not in combined]
    if absent_copy:
        raise AssertionError(f"Required scientific-state copy is absent: {absent_copy}")

    print(
        "FRONTEND CONTRACT TEST PASSED: "
        f"{len(collector.ids)} unique IDs, {len(referenced_ids)} JS references"
    )


if __name__ == "__main__":
    main()
