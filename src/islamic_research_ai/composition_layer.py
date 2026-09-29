"""`AnswerComposer` -- WP-2 of `CONVERSATIONAL_QUALITY_TECHNICAL_PLAN_V3.md`.

    retrieval -> evidence -> [ GroqSynthesizer ] -> claims (strict, evidence-bound, audited)
                                                     |
                                              [ AnswerComposer ]        <- this module
                                                     |
                                            [ AntiFabricationGate ]     <- anti_fabrication.py
                                                     |
                                          narrative + claims + sources

The composer turns already-verified, evidence-bound claims into connected reader-facing prose.
It is deliberately given the claims' own text and evidence *labels* (e.g. "Quran 2:155", "Ibn
Kathir tafsir on 2:155") rather than raw canonical text, so it has nothing to paraphrase from
in the first place -- the `AntiFabricationGate` is the second, independent layer of defense,
not the only one.

On a gate failure the composer gets exactly one bounded repair, naming the offending rule and
span (mirroring the retry pattern `GroqSynthesizer` already uses). On a second failure it
returns `None`: the caller renders claims and sources with no narrative. That is a deliberate
degrade to pre-WP-2 behaviour, never a fabrication -- see Part 2 and Part 3b of the plan.
"""

from __future__ import annotations

import json
from collections.abc import Callable

from islamic_research_ai.anti_fabrication import (
    AntiFabricationGate,
    FabricationRule,
    FabricationViolation,
)
from islamic_research_ai.chat_routing import TurnChatRouteState
from islamic_research_ai.contracts import (
    AnswerStatus,
    EvidenceSet,
    NarrativeDraftV1,
    NarrativeSegmentV1,
    ProviderCompletion,
    RenderedClaimBlockV1,
    RenderedNarrativeBlockV1,
)
from islamic_research_ai.local_inference import local_completion_attribution
from islamic_research_ai.ports import (
    ChatModelPort,
    ProviderFailureDiagnostic,
    ProviderUnavailableError,
    StructuredOutputValidationError,
)
from islamic_research_ai.providers.groq import GroqError
from islamic_research_ai.routing import (
    DEFAULT_GROQ_ROLE_DEPLOYMENT,
    ProviderRouteError,
    ProviderRoutePolicy,
)


class AnswerComposer:
    SYSTEM_PROMPT_AR = """أنت تكتب الإجابة التي يقرأها الباحث في أداة بحث قرآنية. لديك ادعاءات
تم التحقق منها بالفعل مقابل مصادر مسترجعة، ولكل ادعاء تسميات مرجعية (وليس النص الكنسي نفسه).
أعد تنظيم هذه الادعاءات في إجابة واضحة ومترابطة بالعربية الفصحى. يمكنك كتابة جمل تمهيدية
(FRAMING) وانتقالية (TRANSITION) وخاتمة (SUMMARY)، وشرح كيف ترتبط الأدلة المستشهد بها بالسؤال،
وقسم محدود يوضح الحدود إن لم يغط كل جوانب السؤال (LIMITATION). يجب ألا تقتبس أو تعيد صياغة أي
نص قرآني -- التطبيق يعرض النص الكنسي بشكل منفصل. يجب ألا تذكر أي مرجع آية أو سورة أو تفسير أو
عالم أو حديث غير موجود في الادعاءات أو تسمياتها المرجعية. عند اختلاف العلماء، اعرض الآراء دون
ترجيح أحدها. أرجع فقط JSON مطابقًا للمخطط narrative_draft_v1."""

    SYSTEM_PROMPT_EN = """You are writing the reader-facing answer for a Quran research
assistant. You are given verified claims already checked against retrieved sources, each with
reference labels (not the canonical text itself). Reorganise these claims into a clear,
connected answer in English. You may write framing sentences (FRAMING), transitions
(TRANSITION), a closing summary (SUMMARY), and explain how the cited evidence relates to the
question, plus one bounded limitation segment if part of the question is not covered
(LIMITATION). You must not quote or paraphrase any Quranic wording -- the application renders
canonical text separately. You must not mention any verse reference, surah, tafsir, scholar, or
hadith absent from the supplied claims or their reference labels. On questions where recognised
schools differ, present the views without endorsing one. Return only JSON matching the
narrative_draft_v1 schema."""

    def __init__(
        self,
        client: ChatModelPort,
        gate: AntiFabricationGate | None = None,
        *,
        local_client: ChatModelPort | None = None,
        route_validator: Callable[[ProviderCompletion], None] | None = None,
        local_route_validator: Callable[[ProviderCompletion], None] | None = None,
        max_tokens: int = 2000,
    ) -> None:
        self.client = client
        self.local_client = local_client
        self.gate = gate or AntiFabricationGate()
        self.route_validator = route_validator or (
            lambda completion: ProviderRoutePolicy.require_configured_route(
                completion, role_deployment=DEFAULT_GROQ_ROLE_DEPLOYMENT
            )
        )
        self.local_route_validator = local_route_validator or (lambda completion: None)
        self.max_tokens = max_tokens

    async def compose(
        self,
        *,
        question: str,
        response_language: str,
        coverage_status: AnswerStatus,
        claims: list[RenderedClaimBlockV1],
        evidence: EvidenceSet,
        route_state: TurnChatRouteState | None = None,
    ) -> tuple[RenderedNarrativeBlockV1 | None, list[ProviderCompletion], dict[str, int]]:
        """Returns narrative, completions, and durable-safe rejection counts.

        `narrative` is `None` on a double gate failure.  The counts name rules only;
        they intentionally omit model output and offending spans from storage and the
        client response.

        The composer itself only ever sees `claims` (already-verified text plus reference
        *labels*, never raw canonical text) -- but `AntiFabricationGate` checks the composer's
        output against the full, real `evidence` retrieved this turn, independently of what the
        composer was told. That is the second, independent layer of defense: even if the
        composer somehow produced a real verse or attribution it was never given, the gate
        still only accepts it if it is genuinely present in this turn's evidence.
        """

        if not claims:
            return None, [], {}

        claim_payload = [
            {
                "claim_id": claim.claim_id,
                "text": claim.text,
                "evidence_labels": [citation.display_label for citation in claim.citations],
            }
            for claim in claims
        ]
        claim_evidence_ids = {
            claim.claim_id: [citation.evidence_id for citation in claim.citations]
            for claim in claims
        }

        completions: list[ProviderCompletion] = []
        gate_rejections_by_rule: dict[str, int] = {}
        repair_note: str | None = None
        for _attempt in range(2):
            payload = {
                "question": question,
                "response_language": response_language,
                "coverage_status": coverage_status.value,
                "claims": claim_payload,
            }
            prompt = json.dumps(payload, ensure_ascii=False)
            if repair_note:
                prompt += f"\nThis is the only repair. {repair_note}"
            system_prompt = (
                self.SYSTEM_PROMPT_AR if response_language == "ar" else self.SYSTEM_PROMPT_EN
            )
            draft, completion = await self._complete_with_route(
                route_state=route_state,
                response_model=NarrativeDraftV1,
                schema_name="narrative_draft_v1",
                system_prompt=system_prompt,
                user_prompt=prompt,
            )
            completions.append(completion)

            narrative = self._to_rendered(draft, claim_evidence_ids)
            narrative_text = "\n".join(segment.text for segment in narrative.segments)
            violations = self.gate.check(narrative_text=narrative_text, evidence=evidence)
            if not violations:
                return narrative, completions, gate_rejections_by_rule
            for violation in violations:
                rule = FabricationRule(violation.rule).value
                gate_rejections_by_rule[rule] = gate_rejections_by_rule.get(rule, 0) + 1
            repair_note = self._describe_violations(violations)

        # Second failure: degrade to claims-only. Never render a fabrication (Part 3b, non-
        # negotiable #5).
        return None, completions, gate_rejections_by_rule

    async def _complete_with_route(
        self,
        *,
        route_state: TurnChatRouteState | None,
        response_model: type[NarrativeDraftV1],
        schema_name: str,
        system_prompt: str,
        user_prompt: str,
    ) -> tuple[NarrativeDraftV1, ProviderCompletion]:
        if route_state is None:
            try:
                return await self._complete_primary(
                    response_model=response_model,
                    schema_name=schema_name,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                )
            except (ProviderUnavailableError, GroqError, StructuredOutputValidationError) as exc:
                diagnostic = exc.diagnostic if isinstance(exc, ProviderUnavailableError) else None
                raise CompositionProviderFailure(str(exc), diagnostic=diagnostic) from exc
            except ProviderRouteError as exc:
                raise CompositionProviderFailure(str(exc)) from exc
        if route_state.local_selected:
            return await self._complete_local(
                route_state=route_state,
                response_model=response_model,
                schema_name=schema_name,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
            )
        try:
            return await self._complete_primary(
                response_model=response_model,
                schema_name=schema_name,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
            )
        except ProviderUnavailableError as exc:
            if route_state.select_local_after_quota(exc.diagnostic, stage="COMPOSITION"):
                return await self._complete_local(
                    route_state=route_state,
                    response_model=response_model,
                    schema_name=schema_name,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                )
            raise CompositionProviderFailure(str(exc), diagnostic=exc.diagnostic) from exc
        except (GroqError, StructuredOutputValidationError, ProviderRouteError) as exc:
            raise CompositionProviderFailure(str(exc)) from exc

    async def _complete_primary(
        self,
        *,
        response_model: type[NarrativeDraftV1],
        schema_name: str,
        system_prompt: str,
        user_prompt: str,
    ) -> tuple[NarrativeDraftV1, ProviderCompletion]:
        try:
            draft, completion = await self.client.complete(
                response_model=response_model,
                schema_name=schema_name,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_tokens=self.max_tokens,
            )
            self.route_validator(completion)
            return draft, completion
        except (ProviderUnavailableError, GroqError, StructuredOutputValidationError):
            raise

    async def _complete_local(
        self,
        *,
        route_state: TurnChatRouteState,
        response_model: type[NarrativeDraftV1],
        schema_name: str,
        system_prompt: str,
        user_prompt: str,
    ) -> tuple[NarrativeDraftV1, ProviderCompletion]:
        if self.local_client is None:
            raise CompositionProviderFailure("No local composer client is configured")
        try:
            quota_fallback = route_state.selection_stage == "COMPOSITION"
            with local_completion_attribution(quota_fallback=quota_fallback):
                draft, completion = await self.local_client.complete(
                    response_model=response_model,
                    schema_name=schema_name,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    max_tokens=self.max_tokens,
                )
            self.local_route_validator(completion)
            return draft, completion
        except ProviderUnavailableError as exc:
            route_state.record_failure(exc.diagnostic)
            raise CompositionProviderFailure(str(exc), diagnostic=exc.diagnostic) from exc
        except (GroqError, StructuredOutputValidationError, ProviderRouteError) as exc:
            raise CompositionProviderFailure(str(exc)) from exc

    @staticmethod
    def _to_rendered(
        draft: NarrativeDraftV1, claim_evidence_ids: dict[str, list[str]]
    ) -> RenderedNarrativeBlockV1:
        segments: list[NarrativeSegmentV1] = []
        for index, segment in enumerate(draft.segments, start=1):
            # Only claim ids that actually exist survive -- the model cannot manufacture a
            # supporting claim, and therefore cannot manufacture supporting evidence either.
            valid_claim_ids = [
                claim_id
                for claim_id in segment.supporting_claim_ids
                if claim_id in claim_evidence_ids
            ]
            evidence_ids = list(
                dict.fromkeys(
                    evidence_id
                    for claim_id in valid_claim_ids
                    for evidence_id in claim_evidence_ids[claim_id]
                )
            )
            segments.append(
                NarrativeSegmentV1(
                    segment_id=f"N{index}",
                    text=segment.text,
                    kind=segment.kind,
                    supporting_claim_ids=valid_claim_ids,
                    supporting_evidence_ids=evidence_ids,
                )
            )
        return RenderedNarrativeBlockV1(block_type="NARRATIVE", segments=segments)

    @staticmethod
    def _describe_violations(violations: list[FabricationViolation]) -> str:
        parts = [
            f"[{v.rule.value}] remove or fix: {v.offending_span!r} -- {v.detail}"
            for v in violations
        ]
        joined = " ".join(parts)
        return (
            "Fix every issue below and return compact JSON immediately with no analysis. " + joined
        )


class CompositionProviderFailure(RuntimeError):
    """A best-effort narrative provider failed; claims and sources remain valid."""

    def __init__(
        self, safe_detail: str, *, diagnostic: ProviderFailureDiagnostic | None = None
    ) -> None:
        self.safe_detail = safe_detail
        self.diagnostic = diagnostic
        super().__init__(safe_detail)
