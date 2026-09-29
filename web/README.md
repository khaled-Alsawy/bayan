# Quran Research Web POC

This package is the private Quran web vertical slice. It is a thin Next.js application over the
existing FastAPI service; it contains no provider, retrieval, or research-domain logic.

## Runtime

- Node.js 24.19.0 LTS (`.node-version` and `.nvmrc`)
- pnpm 11.19.0
- exact dependencies in `package.json` and `pnpm-lock.yaml`

## Local use

Start the FastAPI service on loopback, then:

```bash
pnpm install --frozen-lockfile
BACKEND_BASE_URL=http://127.0.0.1:8000 pnpm dev
```

Open the development UI at **http://localhost:3000**. The dev script binds to that canonical
origin intentionally; opening the same server as `http://127.0.0.1:3000` can trigger Next.js
development-origin protection and prevent client hydration. The FastAPI service remains loopback
only at `127.0.0.1:8000` behind the same-origin Next.js route.

The UI is Arabic/RTL by default. Switching to English stores the preference in a local cookie and
local storage so the server and browser render the same language on the next visit without a
hydration flash.

The browser uses only `POST /api/research` on the Next.js origin. The route handler validates the
browser request, forwards it to `POST /v1/research`, validates the canonical response, disables
caching, and normalizes failures without returning backend diagnostics.

## Contract workflow

```bash
pnpm contracts:generate
pnpm typecheck
pnpm test
```

`openapi.json` is exported mechanically from FastAPI. `openapi-typescript` creates
`src/contracts/openapi.generated.ts`; bidirectional generic assertions make type-checking fail on
any full-shape request or response drift, and Zod validates both sides of the network boundary.

## Verification

```bash
pnpm lint
pnpm typecheck
pnpm test
pnpm build
pnpm test:e2e
```

The real-browser acceptance additionally runs `pnpm test:e2e:real` while FastAPI is configured for
forced-local Gemma. No deployment or public hosting is configured.
