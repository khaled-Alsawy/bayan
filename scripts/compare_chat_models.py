"""WP-1 (V3 plan, task 6): side-by-side chat-model comparison for the `composer`/`synthesis`
roles -- the roles that produce the reader-facing Arabic and English prose.

Judged criterion (per the plan): Arabic religious register specifically -- fusha, correct
shar'i terminology, no drift into colloquial or into ruling/fatwa language. General
multilingual benchmarks do not capture this, so the prompt set below is domain-specific and
the judging is manual (a general model output is read against that rubric), not scored by
another model.

Candidates, in the plan's stated priority order:
    qwen/qwen3-32b, moonshotai/kimi-k2-instruct, openai/gpt-oss-120b (baseline to beat),
    llama-3.3-70b-versatile

CORRECTION (found by actually running this script against the live Groq account, 2026-09-19):
none of qwen/qwen3-32b, moonshotai/kimi-k2-instruct, or llama-3.3-70b-versatile are available
on this Groq account (HTTP 404 model_not_found for all three). `GET /v1/models` on this account
lists: openai/gpt-oss-120b, openai/gpt-oss-20b, openai/gpt-oss-safeguard-20b, qwen/qwen3.8-27b,
allam-2-7b (SDAIA's Arabic-first model -- the plan's WP-1 item 8 says Arabic-first models are
"none hosted on the current approved route", which this account contradicts), plus audio/guard
models not relevant here. The candidate list below reflects what is actually callable.

Usage:
    .venv/bin/python scripts/compare_chat_models.py

Requires GROQ_API_KEY in the environment (e.g. via scripts/run_api_with_keychain.py's pattern).
Writes a side-by-side artifact to artifacts/e2e/chat-model-comparison.json and prints it.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]

CANDIDATES = [
    "qwen/qwen3.8-27b",
    "openai/gpt-oss-120b",
    "allam-2-7b",
    "openai/gpt-oss-20b",
]

# One Arabic and one English prompt shaped exactly like the WP-2 AnswerComposer's job: turn
# already-verified claims into connected reader-facing prose, without adding new claims.
PROMPTS = {
    "ar_sabr": {
        "system": (
            "أنت تكتب الإجابة التي يقرأها الباحث في أداة بحث قرآنية. لديك ادعاءات تم "
            "التحقق منها بالفعل مقابل مصادر مسترجعة. أعد تنظيمها في إجابة واضحة ومترابطة "
            "بالعربية الفصحى. يمكنك كتابة جمل تمهيدية وانتقالية وخلاصة ختامية، وشرح كيف "
            "ترتبط المقاطع المستشهد بها بالسؤال. يجب ألا تقتبس أو تعيد صياغة النص القرآني "
            "نفسه -- التطبيق يعرض النص الكنسي بشكل منفصل. يجب ألا تذكر أي مرجع آية أو سورة "
            "أو تفسير أو عالم غير موجود في المادة المزودة."
        ),
        "user": (
            "السؤال: ما موقف القرآن من الصبر عند الابتلاء؟\n\n"
            "الادعاءات المتحقق منها:\n"
            "1. في سورة البقرة 2:155-156، يخبر الله أنه سيبتلي الناس بشيء من الخوف والجوع "
            "ونقص الأموال والأنفس والثمرات، ويبشر الصابرين.\n"
            "2. تفسير ابن كثير لهذه الآيات يوضح أن الصبر هنا هو الرضا بقضاء الله دون تسخط، "
            "مع قول 'إنا لله وإنا إليه راجعون' عند المصيبة.\n"
            "3. في سورة الزمر 39:10، يخبر الله أن الصابرين يوفون أجرهم بغير حساب.\n\n"
            "اكتب الإجابة الآن."
        ),
    },
    "en_sabr": {
        "system": (
            "You are writing the reader-facing answer for a Quran research assistant. You are "
            "given verified claims already checked against retrieved sources. Reorganise them "
            "into a clear, connected answer in English. You may write framing sentences, "
            "transitions and a closing summary. You must not quote or paraphrase Quranic "
            "wording -- the application renders canonical text separately. You must not "
            "mention any verse reference, surah, tafsir, or scholar absent from the supplied "
            "material."
        ),
        "user": (
            "Question: What does the Quran say about patience (sabr) during trials?\n\n"
            "Verified claims:\n"
            "1. In Surah Al-Baqarah 2:155-156, God says He will test people with some fear, "
            "hunger, and loss of wealth, lives, and fruits, and gives glad tidings to the "
            "patient.\n"
            "2. Ibn Kathir's commentary on these verses explains that the patience meant here "
            "is contentment with God's decree without resentment, saying 'to God we belong "
            "and to Him we return' upon calamity.\n"
            "3. In Surah Az-Zumar 39:10, God says the patient will be given their reward "
            "without account.\n\n"
            "Write the answer now."
        ),
    },
}


async def _call(client: httpx.AsyncClient, model: str, system: str, user: str) -> dict[str, object]:
    started = time.perf_counter()
    response = await client.post(
        "/chat/completions",
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0.2,
            "max_completion_tokens": 700,
        },
    )
    latency_ms = round((time.perf_counter() - started) * 1000)
    response.raise_for_status()
    body = response.json()
    text = body["choices"][0]["message"]["content"]
    usage = body.get("usage", {})
    return {
        "model": model,
        "latency_ms": latency_ms,
        "completion_tokens": usage.get("completion_tokens"),
        "text": text,
    }


async def main() -> None:
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise SystemExit("GROQ_API_KEY is required")
    results: dict[str, list[dict[str, object]]] = {name: [] for name in PROMPTS}
    async with httpx.AsyncClient(
        base_url="https://api.groq.com/openai/v1",
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=90.0,
    ) as client:
        for prompt_name, prompt in PROMPTS.items():
            for model in CANDIDATES:
                try:
                    result = await _call(client, model, prompt["system"], prompt["user"])
                except httpx.HTTPStatusError as exc:
                    result = {
                        "model": model,
                        "error": f"HTTP {exc.response.status_code}: {exc.response.text[:300]}",
                    }
                results[prompt_name].append(result)
                print(f"--- {prompt_name} :: {model} ---")
                print(result.get("text", result.get("error")))
                print()

    artifact_dir = ROOT / "artifacts" / "e2e"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = artifact_dir / "chat-model-comparison.json"
    artifact_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {artifact_path}")


if __name__ == "__main__":
    asyncio.run(main())
