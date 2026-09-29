import asyncio
import fcntl
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import yaml


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

ROOT = str(Path(__file__).resolve().parents[1])
ACCOUNT = "islamic-research-ai"
for var, svc in {
    "GROQ_API_KEY": "groq-api-key",
    "CLOUDFLARE_API_TOKEN": "cloudflare-api-token",
}.items():
    os.environ[var] = subprocess.run(
        ["/usr/bin/security", "find-generic-password", "-a", ACCOUNT, "-s", svc, "-w"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
os.environ.update(
    {
        "CLOUDFLARE_ACCOUNT_ID": cloudflare_account_id(),
        "PROJECT_OWNER_RISK_OVERRIDE_ACK": "1",
        "LOCAL_CHAT_FORCE": "1",
    }
)

from islamic_research_ai.composition import build_container  # noqa: E402
from islamic_research_ai.contracts import ResearchRequest  # noqa: E402

OUT = sys.argv[1]
LOCK_FILE = open(f"{OUT}.lock", "w")
try:
    fcntl.flock(LOCK_FILE, fcntl.LOCK_EX | fcntl.LOCK_NB)
except BlockingIOError:
    print(f"collector already running for {OUT}", file=sys.stderr)
    raise SystemExit(1) from None

cases = yaml.safe_load(open(f"{ROOT}/qa/cases/semantic_cases.yaml"))
cases = cases if isinstance(cases, list) else cases.get("cases", cases)

done_ids = set()
if os.path.exists(OUT):
    kept = []
    for line in open(OUT, encoding="utf-8", errors="replace"):
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue          # drop a partial final line from a killed run
        done_ids.add(rec["id"])
        kept.append(line)
    with open(OUT, "w") as fh:
        fh.write("\n".join(kept) + ("\n" if kept else ""))
    print(f"resuming: {len(done_ids)} already collected", flush=True)

def record_trace_event(events: list[dict[str, object]]):
    def record(event):
        events.append(event.model_dump(mode="json"))

    return record


async def main():
    c = await build_container()
    done = len(done_ids)
    try:
        with open(OUT, "a") as fh:
            for case in cases:
                if case["id"] in done_ids:
                    continue
                t0 = time.time()
                events: list[dict[str, object]] = []
                rec = {"id": case["id"], "events": events}
                try:
                    resp, _ = await c.service.research(
                        ResearchRequest(message=case["prompt"], locale=case["ui_language"]),
                        on_trace_event=record_trace_event(events),
                    )
                    d = json.loads(resp.model_dump_json())
                    rec["response"] = d
                    rec["retrieved_evidence"] = [
                        b["citation"] for b in (d.get("blocks") or [])
                        if isinstance(b, dict) and isinstance(b.get("citation"), dict)
                    ]
                    rec["elapsed_ms"] = int((time.time()-t0)*1000)
                except Exception as e:
                    rec["error"] = f"{type(e).__name__}: {str(e)[:200]}"
                    rec["elapsed_ms"] = int((time.time()-t0)*1000)
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fh.flush()
                done += 1
                status = rec.get("response", {}).get("status") or rec.get("error", "")[:30]
                print(
                    f"{done}/{len(cases)} {case['id']:34} {status} {rec['elapsed_ms']}ms",
                    flush=True,
                )
    finally:
        await c.close()

try:
    asyncio.run(main())
finally:
    LOCK_FILE.close()
