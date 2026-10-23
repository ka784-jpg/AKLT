"""Check published source/figure hashes and local Markdown targets; stdlib only."""

import hashlib
import json
from pathlib import Path
import re
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]


def check_hash(relative, expected):
    path = (ROOT / relative).resolve()
    if not path.is_relative_to(ROOT) or not path.is_file():
        raise AssertionError(f"Missing or out-of-tree file: {relative}")
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual.lower() != expected.lower():
        raise AssertionError(f"SHA256 mismatch: {relative}")


def main():
    sources = json.loads((ROOT / "SOURCE_MANIFEST.json").read_text(encoding="utf-8-sig"))
    for source in sources:
        check_hash(source["file"], source["packaged_sha256"])
    figures = json.loads((ROOT / "docs/figures/manifest.json").read_text(encoding="utf-8-sig"))["figures"]
    for figure in figures:
        check_hash(figure["pdf"], figure["pdf_sha256"])
        check_hash(figure["preview"], figure["preview_sha256"])

    documents = list(ROOT.glob("*.md")) + list((ROOT / "docs").rglob("*.md")) + list((ROOT / "examples").glob("*.md"))
    link_count = 0
    for document in documents:
        for target in re.findall(r"\]\(([^)\s]+)\)", document.read_text(encoding="utf-8")):
            url = urlsplit(target)
            if url.scheme or url.netloc or not url.path:
                continue
            path = (document.parent / unquote(url.path)).resolve()
            if not path.is_relative_to(ROOT) or not path.exists():
                raise AssertionError(f"Broken local link in {document.relative_to(ROOT)}: {target}")
            link_count += 1
    print(f"Verified {len(sources)} source hashes, {len(figures) * 2} figure hashes, and {link_count} local Markdown paths.")


if __name__ == "__main__":
    main()
