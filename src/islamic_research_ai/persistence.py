from __future__ import annotations

from islamic_research_ai.contracts import (
    EvidenceSourceKind,
    RenderedSourceBlockV1,
    ResearchResponseV1,
)

RESTRICTED_SOURCE_TEXT_REDACTION = "[SOURCE_TEXT_NOT_STORED_POLICY]"


def is_quran_foundation_source_block(block: object) -> bool:
    return (
        isinstance(block, RenderedSourceBlockV1)
        and block.citation.source_kind is EvidenceSourceKind.TAFSIR
        and block.citation.source_revision.source_id == "quran_foundation"
    )


def response_for_durable_storage(response: ResearchResponseV1) -> ResearchResponseV1:
    """Return the canonical response shape with STORE=DENIED source text removed.

    The runtime no longer produces Quran Foundation blocks. This redaction remains for
    historical conversation rows and as defence in depth for their durable representation.
    """

    blocks = [
        block.model_copy(update={"exact_text": RESTRICTED_SOURCE_TEXT_REDACTION})
        if is_quran_foundation_source_block(block)
        else block.model_copy(deep=True)
        for block in response.blocks
    ]
    return response.model_copy(update={"blocks": blocks})


def response_contains_restricted_source_text(response: ResearchResponseV1) -> bool:
    return any(
        is_quran_foundation_source_block(block)
        and block.exact_text != RESTRICTED_SOURCE_TEXT_REDACTION
        for block in response.blocks
    )
