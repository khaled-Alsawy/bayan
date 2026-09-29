from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[2]

_KNOWN_QUL_TAFSIR_ALIASES = frozenset(
    {
        "qul-tafsir-almukhtasar-ar-v1",
        "qul-tafsir-ibnkathir-ar-v1",
        "qul-tafsir-alqurtubi-ar-v1",
        "qul-tafsir-albaghawi-ar-v1",
    }
)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=None, extra="ignore")

    app_environment: str = "development"
    developer_trace_enabled: bool = True
    development_purpose: str = "private_poc"
    source_policy_path: Path = ROOT / "config/source_policies.development.json"
    chat_deployments_path: Path = ROOT / "config/chat_deployments.json"
    project_owner_risk_override_ack: bool = False
    public_access_token: SecretStr | None = None

    database_url: str = "postgresql://127.0.0.1:55432/islamic_research_ai"

    groq_api_key: SecretStr | None = None
    groq_model: str = "openai/gpt-oss-20b"
    groq_base_url: str = "https://api.groq.com/openai/v1"

    local_chat_enabled: bool = True
    local_chat_force: bool = False
    local_chat_base_url: str = "http://127.0.0.1:18080/v1"
    # Direct local-runtime measurements: 256/1024/1555 tokens completed in
    # 4.2/15.1/23.8 seconds (60.8/67.9/65.4 tok/s); a 7,623-token prompt
    # completed in 8.6 seconds.  90 seconds is roughly 3x the observed worst
    # generation, leaving cold-start and contention headroom without allowing
    # a stuck decode to monopolise the sole local slot for four minutes.
    # This is deliberately a code default: Settings does not read dotenv files.
    local_chat_timeout_seconds: float = 90.0
    local_chat_health_timeout_seconds: float = 5.0
    local_chat_max_concurrency: int = 1
    local_chat_max_prompt_characters: int = 100_000
    local_model_manifest_path: Path = ROOT / "config/local-model-gemma4-26b-a4b-q4.json"
    local_model_root: Path = ROOT / ".runtime/models/gemma-4-26b-a4b-it-4bit--0d77464e"

    cloudflare_api_token: SecretStr | None = None
    cloudflare_account_id: str | None = None
    cloudflare_embedding_model: str = "@cf/baai/bge-m3"
    # Switch only after the enriched-vector backfill has completed. Keeping both vectors makes
    # the semantic-retrieval measurement reversible without corpus re-ingestion.
    quran_embedding_variant: Literal["base", "enriched"] = "base"
    quran_enrichment_tafsir_excerpt_characters: int = Field(default=480, ge=1, le=800)
    # QUL's curated hierarchy is a separate retrieval arm, not text merged into verse vectors.
    # It remains off until the full candidate-pool experiment is measured.
    qul_topic_arm_enabled: bool = False
    qul_topic_arm_limit: int = Field(default=20, ge=1, le=50)
    # Offline, this order-only reranker improved the identical fused pool from 26% to 36%
    # overall (high indirection 15% to 25%). It is not a retrieval arm and remains opt-in.
    qul_topic_rerank_enabled: bool = False
    # Workers AI currently exposes bge-reranker-base, not the multilingual v2-m3 name that
    # appeared in the original plan.  It remains opt-in because its Arabic quality has not been
    # evaluated; retrieval keeps its valid fused ordering when this experiment is disabled.
    cloudflare_reranker_enabled: bool = False
    cloudflare_reranker_model: str = "@cf/baai/bge-reranker-base"
    # Forced-local Gemma, topic arm off: replacing concise QF Jalalayn with long, passage-level
    # QUL Ibn Kathir measured 24.0% -> 20.0% anchor hit (high band 16.0% -> 8.0%). Keep this
    # configurable so commentary-shape alternatives can be measured without a code change.
    topic_tafsir_alias: str = "qul-tafsir-almukhtasar-ar-v1"

    tanzil_download_url: str = (
        "https://tanzil.net/pub/download/index.php?quranType=uthmani&outType=txt-2&agree=true"
    )
    quranenc_download_url: str = "https://quranenc.com/downloads/sqlite/english_rwwad.sqlite"
    raw_data_dir: Path = ROOT / "data/raw"
    artifact_dir: Path = ROOT / "artifacts/e2e"

    search_limit: int = 12
    hydrate_limit: int = 12
    topic_anchor_limit: int = 3
    # Ranked retention measured 16.0% against a 24.0% baseline on the 60-case set;
    # the high-indirection band fell to 4.0%, so it remains opt-in for diagnosis only.
    relevance_ranked_retention_enabled: bool = False
    relevance_distinctive_terms_enabled: bool = False
    honest_insufficient_fallback_enabled: bool = False
    relevance_retained_passage_limit: int = Field(default=5, ge=1, le=20)
    # The humanity evidence-term expansion is a controlled measurement arm. It remains off
    # until its effect on the semantic quality set has been measured independently.
    risk_concept_human_subject_terms_enabled: bool = False
    # K1 passed its gate on the 60-case semantic set: acceptable evidence rose from 13/42 to
    # 17/42 (four recovered, none lost); citation hit rose 24.0% to 32.0%; visible-only citation
    # hit rose 22.0% to 26.0%; and mean shown verses rose 5.1 to 7.1. In a blinded, counterbalanced
    # paired comparison, the owner preferred K1 in all 3/3 cases with changed visible passages and
    # judged the 5-to-7 increase useful rather than clutter; the two controls were byte-identical.
    # The fourth recovery, sem_en_work_without_credit, changed only collapsed context and was
    # excluded from that visible-passage comparison rather than counted as a win.
    # The 26.0% visible-only figure is the reader-facing result: the larger 32.0% raw citation gain
    # moved twice as far, so part of it is context behind an expander. Set False to reproduce K0.
    relevance_empty_match_retains_anchors: bool = True

    @model_validator(mode="after")
    def validate_runtime_settings(self) -> Settings:
        """The owner trace is a development instrument, never a production escape hatch."""

        if self.topic_tafsir_alias not in _KNOWN_QUL_TAFSIR_ALIASES:
            raise ValueError("topic_tafsir_alias must name a known QUL Tafsir resource")
        if self.app_environment.casefold() == "production":
            self.developer_trace_enabled = False
        return self


def load_settings() -> Settings:
    return Settings()
