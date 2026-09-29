from __future__ import annotations

import json
import re
import subprocess
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGETS = [ROOT / "artifacts", ROOT / "web/test-results", ROOT / ".runtime/logs"]
KEYCHAIN_SERVICES = [
    "groq-api-key",
    "cloudflare-api-token",
    "quran-foundation-client-id",
    "quran-foundation-client-secret",
]
PATTERNS = {
    "bearer_token": re.compile(rb"(?i)authorization.{0,40}bearer\s+[A-Za-z0-9._~+/=-]{12,}"),
    "qf_auth_header": re.compile(rb"(?i)x-auth-token.{0,100}[A-Za-z0-9._~+/=-]{12,}"),
    "private_user_path": re.compile(rb"/Users/[^/\s]+/(?:Workspace|Desktop|Documents)/"),
}
QF_REDACTIONS = {
    "[SOURCE_TEXT_NOT_STORED_POLICY]",
    "[REDACTED_QF_TAFSIR_TEXT_STORE_DENIED]",
}


def keychain_value(service: str) -> bytes | None:
    process = subprocess.run(
        [
            "security",
            "find-generic-password",
            "-a",
            "islamic-research-ai",
            "-s",
            service,
            "-w",
        ],
        check=False,
        capture_output=True,
    )
    value = process.stdout.strip()
    return value if process.returncode == 0 and value else None


def findings_for_bytes(data: bytes, secrets: dict[str, bytes]) -> list[str]:
    findings = [name for name, value in secrets.items() if value in data]
    findings.extend(name for name, pattern in PATTERNS.items() if pattern.search(data))
    if qf_raw_source_text_present(data):
        findings.append("qf_raw_tafsir_source_text")
    return sorted(set(findings))


def qf_raw_source_text_present(data: bytes) -> bool:
    """Recognize policy-sensitive source blocks in JSON and JSONL artifacts."""

    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return False
    candidates: list[object] = []
    try:
        candidates.append(json.loads(text))
    except json.JSONDecodeError:
        for line in text.splitlines():
            try:
                candidates.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    def walk(value: object) -> bool:
        if isinstance(value, list):
            return any(walk(item) for item in value)
        if not isinstance(value, dict):
            return False
        citation = value.get("citation")
        if (
            value.get("block_type") == "SOURCE"
            and isinstance(citation, dict)
            and citation.get("source_kind") == "TAFSIR"
        ):
            revision = citation.get("source_revision")
            if (
                isinstance(revision, dict)
                and revision.get("source_id") == "quran_foundation"
                and value.get("exact_text") not in QF_REDACTIONS
            ):
                return True
        return any(walk(item) for item in value.values())

    return any(walk(value) for value in candidates)


def main() -> None:
    secrets = {
        service: value
        for service in KEYCHAIN_SERVICES
        if (value := keychain_value(service)) is not None
    }
    findings: list[tuple[str, list[str]]] = []
    files_scanned = 0
    zip_entries_scanned = 0
    for target in TARGETS:
        if not target.exists():
            continue
        for path in target.rglob("*"):
            if not path.is_file():
                continue
            files_scanned += 1
            try:
                data = path.read_bytes()
            except OSError:
                continue
            matched = findings_for_bytes(data, secrets)
            if target == ROOT / ".runtime/logs":
                # Launch/runtime warnings may identify the local executable path;
                # keep the operational-log audit focused on credentials and
                # restricted provider content.
                matched = [item for item in matched if item != "private_user_path"]
            if matched:
                findings.append((str(path.relative_to(ROOT)), matched))
            if not zipfile.is_zipfile(path):
                continue
            with zipfile.ZipFile(path) as archive:
                for member in archive.infolist():
                    if member.is_dir():
                        continue
                    zip_entries_scanned += 1
                    try:
                        member_data = archive.read(member)
                    except (OSError, RuntimeError, zipfile.BadZipFile):
                        continue
                    matched = findings_for_bytes(member_data, secrets)
                    if matched:
                        findings.append((f"{path.relative_to(ROOT)}::{member.filename}", matched))
    print(f"keychain_values_checked={len(secrets)}")
    print(f"files_scanned={files_scanned}")
    print(f"zip_entries_scanned={zip_entries_scanned}")
    print(f"finding_count={len(findings)}")
    for location, categories in findings:
        print(f"{location}: {','.join(categories)}")


if __name__ == "__main__":
    main()
