# Disposable hosted-provider probes

This directory is limited to pre-implementation feasibility work. It is not
part of the application architecture or a production provider adapter.

## Safety boundary

- Only the pinned provider/model routes in `probe_providers.py` are accepted.
- Inputs are synthetic. No Tanzil, QuranEnc, Quran Foundation, Hadith, tafsir,
  or other religious source text is sent.
- Every live command requires explicit free-only account assertions.
- No paid fallback, gateway routing, credit purchase, or automatic top-up is
  implemented.
- Authorization headers and secret values are never written to result files or
  printed.
- Result artifacts contain synthetic prompts/responses and sanitized metadata.

## Secret storage

The preferred local mechanism is macOS Keychain:

```sh
security find-generic-password -a islamic-research-ai -s groq-api-key -w
security find-generic-password -a islamic-research-ai -s cloudflare-api-token -w
```

The probe reads `GROQ_API_KEY`, `CLOUDFLARE_API_TOKEN`, and
`CLOUDFLARE_ACCOUNT_ID` from its process environment. A launcher may populate
the two token variables from Keychain for one process. Never place token values
in this repository.

Copy `.env.feasibility.example` to the ignored `.env.feasibility` only after
visually verifying the provider account controls. The script does not parse the
file automatically; export the non-secret values and assertions explicitly.

## Commands

```sh
python3 scripts/feasibility/probe_providers.py groq --repetitions 20 --max-tokens 1000 --pace-seconds 14
python3 scripts/feasibility/probe_providers.py groq-one-repair --mixed-repetitions 10 --max-tokens 1000 --pace-seconds 14
python3 scripts/feasibility/probe_providers.py cloudflare-embeddings
python3 scripts/feasibility/probe_providers.py cloudflare-chat --repetitions 20
```

Each command writes a timestamped JSON result under
`artifacts/feasibility/results/` and prints only a compact non-secret summary.

The paced Groq command deliberately takes about 28 minutes. It stays below the
documented Free 8,000 TPM limit when the prompt and requested completion budget
are combined. Do not shorten the pace merely to finish the evidence run sooner:
a provider 429 is evidence of rate limiting, not a schema-capability result.

`groq-one-repair` is a disposable test of the architecture-approved repair
boundary only. Each case receives an initial planner call, local validation,
and at most one repair call. The repair carries the same semantic context and
validation errors under the same native schema; it never includes raw source
content, recursively retries, or invokes a fallback provider.
