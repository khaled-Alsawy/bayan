"""WP-1 (V3 plan): load the per-role Groq chat deployment map.

Moves chat-deployment identity out of `routing.py` constants and into
`config/chat_deployments.json`, so changing a role's model is a config edit, not a code change.
The route itself (ZDR, no training, no subprocessors, no paid route, no fallback) stays a code
assertion in `routing.ProviderRoutePolicy.require_configured_route` -- only model identity and
token budgets are data.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from islamic_research_ai.routing import GroqRoleDeployment


@dataclass(frozen=True)
class RoleDeploymentConfig:
    role: str
    model_id: str
    max_output_tokens: int
    # Only populated for "synthesis", which needs a larger budget on the tafsir path (F-01).
    max_output_tokens_tafsir: int | None = None

    @property
    def groq_role_deployment(self) -> GroqRoleDeployment:
        return GroqRoleDeployment(role=self.role, model_id=self.model_id)


def load_role_deployments(path: Path) -> dict[str, RoleDeploymentConfig]:
    data = json.loads(path.read_text(encoding="utf-8"))
    result: dict[str, RoleDeploymentConfig] = {}
    for role, raw in data["roles"].items():
        default_tokens = raw.get("max_output_tokens") or raw.get("max_output_tokens_quran")
        if default_tokens is None:
            raise ValueError(f"chat_deployments.json role {role!r} has no token budget")
        result[role] = RoleDeploymentConfig(
            role=role,
            model_id=raw["model_id"],
            max_output_tokens=default_tokens,
            max_output_tokens_tafsir=raw.get("max_output_tokens_tafsir"),
        )
    return result
