from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "artifacts/qa/tested-state.json"
INCLUDE_ROOTS = [
    ROOT / "src",
    ROOT / "tests",
    ROOT / "config",
    ROOT / "migrations",
    ROOT / "scripts",
    ROOT / "web/src",
    ROOT / "web/tests",
    ROOT / "web/e2e",
]
INCLUDE_FILES = [
    ROOT / "pyproject.toml",
    ROOT / "README.md",
    ROOT / "LOCAL_RUNBOOK.md",
    ROOT / "ISLAMIC_RESEARCH_AI_V2_1_CONVERSATION_HISTORY_ADDENDUM.md",
    ROOT / "QA_REMEDIATION_REPORT.md",
    ROOT / "CHAT_HISTORY_IMPLEMENTATION_REPORT.md",
    ROOT / "web/package.json",
    ROOT / "web/pnpm-lock.yaml",
    ROOT / "web/next.config.ts",
    ROOT / "web/playwright.config.ts",
    ROOT / "web/tsconfig.json",
    ROOT / "web/openapi.json",
]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    paths = set(INCLUDE_FILES)
    for source in INCLUDE_ROOTS:
        paths.update(
            path
            for path in source.rglob("*")
            if path.is_file()
            and "node_modules" not in path.parts
            and "__pycache__" not in path.parts
        )
    records = [
        {
            "path": str(path.relative_to(ROOT)),
            "sha256": digest(path),
            "size": path.stat().st_size,
            "modified_ns": path.stat().st_mtime_ns,
        }
        for path in sorted(paths)
    ]
    manifest_material = "".join(f"{item['sha256']}  {item['path']}\n" for item in records).encode()
    payload = {
        "captured_at": datetime.now(UTC).isoformat(),
        "git_revision": None,
        "git_revision_note": "Workspace is not a Git checkout.",
        "manifest_sha256": hashlib.sha256(manifest_material).hexdigest(),
        "file_count": len(records),
        "files": records,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: payload[key] for key in ("manifest_sha256", "file_count")}))


if __name__ == "__main__":
    main()
