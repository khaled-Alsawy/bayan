from __future__ import annotations

import re

from islamic_research_ai.contracts import (
    DeterministicResolutionContextV1,
    ScopeDecisionV1,
    ScopeReason,
    ScopeSignal,
    ScopeStatus,
    no_clarification,
)
from islamic_research_ai.retrieval.normalization import normalize_arabic, normalize_translation

_FIQH = re.compile(
    r"\b(fatwa|halal(?:\s+or\s+haram)?|haram|legal ruling|binding ruling|"
    r"divorce ruling|inheritance ruling|madhhab ruling)\b|"
    r"(?:فتوى|افتني|أفتني|حلال|حرام|حكم(?:ا|ً)?\s+شرعي(?:ا|ًا)?|حكم\s+ملزم)",
    re.IGNORECASE,
)
_PERSONAL = re.compile(
    r"\b(should i|must i|for me|tell me what to do|what must i personally do)\b|"
    r"(?:هل\s+(?:يجوز\s+لي|اترك|أترك|افعل|أفعل)|أفتني|افتني|"
    r"أعطني\s+حكما?\s+ملزما?|لي\s+شخصي(?:ا|ًا)|شخصي(?:ا|ًا))",
    re.IGNORECASE,
)
_INJECTION = re.compile(
    r"\b(ignore|disregard|override)\b.{0,80}\b(instructions?|system prompt|rules?)\b|"
    r"\b(reveal|show|print|leak)\b.{0,60}\b(api keys?|secrets?|system prompt)\b|"
    r"(?:تجاهل|تخط).{0,80}(?:التعليمات|الأوامر)|"
    r"(?:اكشف|اظهر|أظهر).{0,60}(?:المفاتيح|الأسرار|تعليمات النظام)",
    re.IGNORECASE,
)
_PLAIN_OUT_OF_DOMAIN = re.compile(
    r"\b(world cup|fifa|football score|weather|stock price|flight status|election result|"
    r"president of|capital of|recipe|computer code|"
    r"(?:write|generate|create)\s+(?:python|javascript|typescript|java|c\+\+|code|a\s+program)"
    r"(?:\s+(?:code|script|program))?)\b|"
    r"(?:كأس العالم|كاس العالم|فيفا|نتيجة المباراة|من فاز بالمباراة|حالة الطقس|"
    r"سعر السهم|سعر الدولار|موعد الرحلة|اكتب(?:\s+لي)?\s+(?:كود|شفرة|برنامج)|"
    r"وصفة طبخ|عاصمة فرنسا)",
    re.IGNORECASE,
)
_EN_PERSONAL_REFERENCE = re.compile(r"\b(?:i|me|my|mine)\b", re.IGNORECASE)
_EN_CONDITIONAL = re.compile(r"\b(?:if|when)\b", re.IGNORECASE)
_EN_FUTURE_CERTAINTY = re.compile(r"\b(?:will|guarantee|ensure|promise)\b", re.IGNORECASE)
_EN_DEVOTIONAL_ACT = re.compile(
    r"\b(?:recit\w*|read\w*)\b.{0,80}\b(?:quran|scripture)\b", re.IGNORECASE
)
_AR_PERSONAL_REFERENCE = re.compile(
    r"(?:^|\s)(?:لي|أنا|انا|نفسي)(?:\s|$)|\b(?!ال)[\u0600-\u06ff]{3,}(?:ي|تي)\b"
)
_AR_PREDICTION_RELATION = re.compile(r"(?:اضمن|تضمن|يضمن|سوف|هت\w*|سي\w*|لو|إذا|اذا|إن|ان)")
_AR_DEVOTIONAL_ACT = re.compile(r"(?:اقر\w*|قري\w*|تلا\w*)")
_AR_PERSONAL_ACTION_FORM = re.compile(
    # First-person Arabic verbs have an initial hamza/alif.  Requiring an attached
    # personal reference separately avoids treating a textual topic such as
    # ``ما موقف القرآن من الطلاق؟`` as a request for advice.  Excluding ``ال`` is
    # important: it is the definite article, not the first-person verb prefix.
    r"(?:^|[\s،,;:])(?:[أإا](?!ل)[\u0621-\u064a]{2,})(?=$|[\s؟?،,.;:])"
)
_AR_REALTIME_LOOKUP_FORM = re.compile(
    # A temporal interrogative plus a relative date requests live schedule data,
    # irrespective of the subject.  It is not a list of religious keywords.
    r"(?:امتي|متى|الساعة|وقت).{0,80}(?:اليوم|النهارده|بكره|غدا)",
    re.IGNORECASE,
)


def _is_arabic_personal_action_request(message: str, normalized: str) -> bool:
    """Recognize the grammatical shape of a personal action/ruling request.

    Arabic commonly attaches the personal relationship or possession to a noun
    (``مراتي``, ``زوجي``, ``زواجي``).  A question which also contains a
    first-person action form asks what the writer should do in that personal
    situation.  This deliberately relies on morphology and sentence shape, not
    on a list of marriage, divorce, or dialect-specific keywords.
    """

    has_question_form = "؟" in message or "?" in message or normalized.startswith("هل ")
    return (
        has_question_form
        and bool(_AR_PERSONAL_REFERENCE.search(normalized))
        and bool(_AR_PERSONAL_ACTION_FORM.search(normalized))
    )


def _is_arabic_realtime_lookup(normalized: str) -> bool:
    """Reject live schedule/date lookups rather than guessing from Qur'an sources."""

    return bool(_AR_REALTIME_LOOKUP_FORM.search(normalized))


def _is_personal_devotional_future_prediction(message: str, normalized: str) -> bool:
    """Separate a textual question from a requested personal worldly prediction.

    This requires the asker, a conditional/certainty relation, and a devotional act. It
    intentionally does not classify themes such as success, wealth, or the
    hereafter; those are valid subjects of Quran research when asked as textual questions.
    """

    english = (
        bool(_EN_PERSONAL_REFERENCE.search(message))
        and bool(_EN_CONDITIONAL.search(message))
        and bool(_EN_FUTURE_CERTAINTY.search(message))
        and bool(_EN_DEVOTIONAL_ACT.search(message))
    )
    # Arabic possession is often attached to the noun (``مشروعي``), rather than a
    # stand-alone pronoun.  Together with a promise/condition and devotional act,
    # that is a request to predict the asker's outcome, not a question about a text.
    arabic = (
        bool(_AR_PERSONAL_REFERENCE.search(normalized))
        and bool(_AR_PREDICTION_RELATION.search(normalized))
        and bool(_AR_DEVOTIONAL_ACT.search(normalized))
    )
    return english or arabic


class ScopePolicy:
    def precheck(
        self, message: str, resolution: DeterministicResolutionContextV1
    ) -> ScopeDecisionV1:
        normalized = normalize_arabic(normalize_translation(message))
        signals = set(resolution.deterministic_scope_signals)
        if _FIQH.search(message) or _FIQH.search(normalized):
            signals.add(ScopeSignal.REQUESTS_FIQH_RULING)
        if _PERSONAL.search(message) or _PERSONAL.search(normalized):
            signals.add(ScopeSignal.REQUESTS_PERSONAL_RELIGIOUS_DECISION)
        if _INJECTION.search(message) or _INJECTION.search(normalized):
            signals.add(ScopeSignal.OUTSIDE_SUPPORTED_RESEARCH)
        if _PLAIN_OUT_OF_DOMAIN.search(message) or _PLAIN_OUT_OF_DOMAIN.search(normalized):
            signals.add(ScopeSignal.OUTSIDE_SUPPORTED_RESEARCH)
        if _is_personal_devotional_future_prediction(message, normalized):
            signals.add(ScopeSignal.OUTSIDE_SUPPORTED_RESEARCH)
        if _is_arabic_personal_action_request(message, normalized):
            signals.add(ScopeSignal.REQUESTS_PERSONAL_RELIGIOUS_DECISION)
        if _is_arabic_realtime_lookup(normalized):
            signals.add(ScopeSignal.OUTSIDE_SUPPORTED_RESEARCH)
        if signals & {
            ScopeSignal.REQUESTS_FIQH_RULING,
            ScopeSignal.REQUESTS_PERSONAL_RELIGIOUS_DECISION,
        }:
            return ScopeDecisionV1(
                schema_version="1",
                status=ScopeStatus.UNSUPPORTED,
                reason=ScopeReason.FIQH_OR_FATWA_NOT_SUPPORTED,
                signals=signals,
                clarification=no_clarification(),
                user_message_key="scope.unsupported_fiqh_or_personal_decision",
            )
        if signals:
            return ScopeDecisionV1(
                schema_version="1",
                status=ScopeStatus.UNSUPPORTED,
                reason=ScopeReason.OUTSIDE_SUPPORTED_RESEARCH,
                signals=signals,
                clarification=no_clarification(),
                user_message_key="scope.outside_supported_research",
            )
        return ScopeDecisionV1(
            schema_version="1",
            status=ScopeStatus.ALLOWED,
            reason=ScopeReason.SUPPORTED,
            signals=set(),
            clarification=no_clarification(),
            user_message_key="scope.supported",
        )
