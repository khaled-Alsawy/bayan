# Web runtime pin

- Runtime: Node.js 24.19.0 LTS (Krypton)
- Package manager: pnpm 11.19.0
- Framework: Next.js 16.3.5 / React 19.2.8
- TypeScript: 5.9.3, strict mode
- Lockfile: `pnpm-lock.yaml`
- Lockfile SHA-256: `4d58fbdce5316106a7199293a40d2aa69e475bb60d488a3f9f25986b23e929a9`
- Implementation record date: 2026-09-16

The runtime pin is recorded in `.node-version`, `.nvmrc`, `package.json#engines`, and the package
manager field. Node.js 24 was an active LTS line at implementation time. All direct dependencies
use exact versions; transitive dependencies are fixed by the lockfile.
