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
    "GROQ_API_KEY": "groq-api-key",
    "CLOUDFLARE_API_TOKEN": "cloudflare-api-token",
}


def keychain_secret(service: str) -> str:
    result = subprocess.run(
        [
            "/usr/bin/security",
            "find-generic-password",
            "-a",
            KEYCHAIN_ACCOUNT,
            "-s",
            service,
            "-w",
        ],
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
    # Direct local measurements reached 60.8/67.9/65.4 tok/s across 256/1024/1555-token
    # generations, and a 7,623-token prompt took 8.6s. `.env` files are never read
    # (env_file=None), so keep the launcher aligned with the 90-second code default rather
    # than accidentally restoring the obsolete 240-second override.
    environment.setdefault("LOCAL_CHAT_TIMEOUT_SECONDS", "90")
    executable = ROOT / ".venv/bin/islamic-research-api"
    os.execve(executable, [str(executable)], environment)


if __name__ == "__main__":
    main()
