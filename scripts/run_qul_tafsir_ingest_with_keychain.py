"""Run the QUL tafsir ingestion (WP-0, V3 plan) with credentials from macOS Keychain.

Mirrors `scripts/run_api_with_keychain.py`: secrets are read directly into this process's
environment and never printed, echoed, or written to a file. Requires the QUL corpus already
downloaded to `data/raw/qul/extracted/` (see the acquisition note in the V3 plan, WP-0) -- this
script ingests it, it does not fetch it.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path


def cloudflare_account_id() -> str:
    """Account IDs stay out of source control: env first, then the gitignored .env.feasibility."""

    value = os.environ.get("CLOUDFLARE_ACCOUNT_ID", "").strip()
    if value:
        return value
    local = Path(__file__).resolve().parents[1] / ".env.feasibility"
    if local.is_file():
        for line in local.read_text(encoding="utf-8").splitlines():
            if line.startswith("CLOUDFLARE_ACCOUNT_ID="):
                return line.split("=", 1)[1].strip()
    raise RuntimeError("Set CLOUDFLARE_ACCOUNT_ID or add it to .env.feasibility")

ROOT = Path(__file__).resolve().parents[1]
KEYCHAIN_ACCOUNT = "islamic-research-ai"
KEYCHAIN_SERVICES = {
    "CLOUDFLARE_API_TOKEN": "cloudflare-api-token",
}


def keychain_secret(service: str) -> str:
    result = subprocess.run(
        ["/usr/bin/security", "find-generic-password", "-a", KEYCHAIN_ACCOUNT, "-s", service, "-w"],
        check=True,
        capture_output=True,
        text=True,
    )
    value = result.stdout.strip()
    if not value:
        raise RuntimeError(f"Required Keychain service is empty: {service}")
    return value


def main() -> None:
    environment = os.environ.copy()
    for variable, service in KEYCHAIN_SERVICES.items():
        environment[variable] = keychain_secret(service)
    environment["CLOUDFLARE_ACCOUNT_ID"] = cloudflare_account_id()
    environment.setdefault("PROJECT_OWNER_RISK_OVERRIDE_ACK", "1")
    executable = ROOT / ".venv/bin/islamic-research-ingest-qul-tafsir"
    os.execve(executable, [str(executable)], environment)


if __name__ == "__main__":
    main()
