from __future__ import annotations

from typing import Literal, Protocol, TypeVar

from pydantic import BaseModel, ConfigDict, Field

from islamic_research_ai.contracts import (
    EvidenceItem,
    ProviderCompletion,
    QuranSearchCandidateV1,
    VerseKey,
)

T = TypeVar("T", bound=BaseModel)


class StructuredOutputValidationError(RuntimeError):
    """The provider generated JSON that did not satisfy the requested schema."""


class ProviderFailureDiagnostic(BaseModel):
    """Privacy-safe internal metadata for a required-provider availability failure."""

    model_config = ConfigDict(extra="forbid")

    provider_id: str
    category: Literal["RATE_LIMITED", "QUOTA_EXHAUSTED", "UNAVAILABLE", "TIMEOUT"]
    http_status: int | None
    retry_after_ms: int | None = Field(default=None, ge=0)
    rate_limit_reset_tokens_ms: int | None = Field(default=None, ge=0)
    quota_limit_tokens: int | None = Field(default=None, ge=0)
    quota_used_tokens: int | None = Field(default=None, ge=0)
    quota_requested_tokens: int | None = Field(default=None, ge=0)
    request_id: str | None
    safe_detail: str


class ProviderUnavailableError(RuntimeError):
    """A required provider cannot serve the request at this time."""

    def __init__(self, diagnostic: ProviderFailureDiagnostic) -> None:
        self.diagnostic = diagnostic
        super().__init__(
            f"{diagnostic.provider_id} {diagnostic.category.lower()}: {diagnostic.safe_detail}"
        )


class ChatModelPort(Protocol):
    async def complete(
        self,
        *,
        response_model: type[T],
        schema_name: str,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 1800,
    ) -> tuple[T, ProviderCompletion]: ...


class EmbeddingModelPort(Protocol):
    model: str
    deployment_id: str
    dimension: int

    async def embed(self, texts: list[str]) -> tuple[list[list[float]], ProviderCompletion]: ...


class RerankerPort(Protocol):
    """Scores query/passage pairs without ever producing source text."""

    model: str
    deployment_id: str

    async def rerank(
        self, *, query: str, passages: list[str]
    ) -> tuple[list[float], ProviderCompletion]: ...


class QuranSourcePort(Protocol):
    async def search(self, query: str, *, limit: int) -> list[QuranSearchCandidateV1]: ...

    async def hydrate_candidates(
        self, candidates: list[QuranSearchCandidateV1], *, limit: int
    ) -> list[EvidenceItem]: ...

    async def hydrate_references(
        self, verse_keys: list[VerseKey], *, limit: int
    ) -> list[EvidenceItem]: ...


class TafsirSourcePort(Protocol):
    async def fetch_passages(
        self,
        verse_keys: list[VerseKey],
        *,
        resource_aliases: list[str],
        character_budget: int,
    ) -> list[EvidenceItem]: ...
