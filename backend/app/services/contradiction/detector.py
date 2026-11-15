import logging
import re
import uuid
from typing import Any, Dict, List, Optional, Set, Tuple

from app.models.schemas import (
    ContradictionItem,
    ContradictionType,
    DocumentMetadata,
    Entity,
    Event,
    NLILabel,
    ReviewStatus,
)
from app.services.ingestion.preprocessor import split_into_sentences
from app.services.llm_provider import analysis_model_json

logger = logging.getLogger(__name__)

# Key thematic conflict word pairs
CONFLICT_PAIRS = [
    (r"\b(unlocked|open|left open)\b", r"\b(locked|secured|shut)\b"),
    (r"\b(authorized|permitted|allowed)\b", r"\b(unauthorized|forbidden|prohibited|not allowed)\b"),
    (r"\b(alone|by myself|solo)\b", r"\b(accompanied|with|together with)\b"),
    (r"\b(inside|entered|accessed|swiped|authenticated|logged in)\b", r"\b(outside|never entered|did not enter|never inside|not inside|away|offsite)\b"),
    (r"\b(present|there|at the site)\b", r"\b(absent|at home|offsite|elsewhere|never there)\b"),
    (r"\b(downloaded|copied|extracted)\b", r"\b(did not download|never copied|no transfer)\b"),
    (r"\b(working|active|online)\b", r"\b(broken|disabled|offline|malfunctioning)\b"),
]

NEGATION_WORDS = {"not", "never", "no", "didn't", "wasn't", "couldn't", "denied", "denies", "refused", "neither"}

TRANSCRIPT_HEADER = re.compile(
    r"^\[\d{1,2}:\d{2}(?::\d{2})?\s*(?:AM|PM)?\]\s*[A-Z][A-Z ._'’-]{0,60}:\s*",
    re.IGNORECASE,
)
CLOCK_TIME = re.compile(r"\b(\d{1,2}):(\d{2})\s*(am|pm)?\b", re.IGNORECASE)
CELSIUS = re.compile(r"(?<![\w.])(?:(minus|plus|[-+−])\s*)?(\d+(?:\.\d+)?)\s*(?:°\s*C\b|Celsius\b|C\b)", re.IGNORECASE)
CONTINUOUS_TEMPERATURE = re.compile(r"\b(?:maintained|stayed|remained|held steady)\b.*\b(?:throughout|continuously|entire)\b")
TEMPERATURE_RISE = re.compile(r"\b(?:rose|climbed|warmed(?: up)?|increased)\s+to\b")
MONITORING_DATE = re.compile(
    r"\b(?:january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2}\b"
)
WINDOW = re.compile(r"\bfrom (\d{1,2}:\d{2}\s*(?:am|pm)) to (\d{1,2}:\d{2}\s*(?:am|pm)|noon|midnight)\b")


def clock_minutes(match: re.Match) -> int:
    hour, minute, meridiem = match.groups()
    hours = int(hour)
    if meridiem:
        hours = hours % 12 + (12 if meridiem.lower() == "pm" else 0)
    return hours * 60 + int(minute)


def celsius_value(match: re.Match) -> float:
    sign, value = match.groups()
    return float(value) * (-1 if (sign or "").lower() in {"minus", "-", "−"} else 1)


def temperature_reading_in_window(baseline: str, reading: str) -> bool:
    dates = MONITORING_DATE.findall(baseline)
    window = WINDOW.search(baseline)
    times = list(CLOCK_TIME.finditer(reading))
    if len(dates) != 1 or dates != MONITORING_DATE.findall(reading) or not window or len(times) != 1:
        return False
    start = CLOCK_TIME.fullmatch(window[1])
    end = CLOCK_TIME.fullmatch(window[2])
    if not start or not times[0][3]:
        return False
    end_minutes = clock_minutes(end) if end else {"noon": 720, "midnight": 1440}[window[2]]
    return clock_minutes(start) <= clock_minutes(times[0]) <= end_minutes


class ContradictionDetector:
    def __init__(self):
        self._nli_pipeline = None
        self._nli_model_name = "roberta-large-mnli"
        self._attempted_pipeline = False

    def _get_nli_pipeline(self):
        if self._attempted_pipeline:
            return self._nli_pipeline
        self._attempted_pipeline = True
        try:
            from transformers import pipeline
            self._nli_pipeline = pipeline("text-classification", model="roberta-large-mnli")
            logger.info("Loaded RoBERTa-MNLI pipeline successfully.")
        except Exception as e:
            logger.info(f"Transformers RoBERTa-MNLI unavailable ({e}). Using deterministic rule/semantic NLI engine.")
            self._nli_pipeline = None
        return self._nli_pipeline

    def evaluate_pair(
        self,
        stmt_a: str,
        stmt_b: str,
        entity_context: str = ""
    ) -> Tuple[NLILabel, ContradictionType, float, str]:
        """
        Backwards-compatible wrapper returning (nli_label, contradiction_type, confidence, explanation).
        """
        label, c_type, conf, expl, _ = self.evaluate_pair_full(stmt_a, stmt_b, entity_context)
        return label, c_type, conf, expl

    def evaluate_pair_full(
        self,
        stmt_a: str,
        stmt_b: str,
        entity_context: str = ""
    ) -> Tuple[NLILabel, ContradictionType, float, str, str]:
        """
        Evaluates premise-hypothesis statement pair.
        Returns (nli_label, contradiction_type, confidence, explanation, detection_engine)
        where detection_engine is one of: 'nli', 'rules', 'llm_verified'.
        """
        stmt_a = TRANSCRIPT_HEADER.sub("", stmt_a.strip())
        stmt_b = TRANSCRIPT_HEADER.sub("", stmt_b.strip())
        a_clean = stmt_a.lower()
        b_clean = stmt_b.lower()

        for statement in (a_clean, b_clean):
            if not statement or statement.endswith("?") or re.search(r"\btell (?:us|me) about\b", statement):
                return (
                    NLILabel.NEUTRAL, ContradictionType.NEUTRAL, 0.50,
                    "At least one passage does not make a factual assertion to compare.", "rules",
                )

        temperatures_a = list(CELSIUS.finditer(a_clean))
        temperatures_b = list(CELSIUS.finditer(b_clean))
        if len(temperatures_a) == len(temperatures_b) == 1:
            temp_a = celsius_value(temperatures_a[0])
            temp_b = celsius_value(temperatures_b[0])
            same_claim = CELSIUS.sub("<temperature>", a_clean) == CELSIUS.sub("<temperature>", b_clean)
            same_entity = entity_context and all(
                re.search(rf"\b{re.escape(entity_context)}\b", statement, re.IGNORECASE)
                for statement in (a_clean, b_clean)
            )
            sustained_vs_rise = same_entity and (
                (CONTINUOUS_TEMPERATURE.search(a_clean) and TEMPERATURE_RISE.search(b_clean) and temp_a < temp_b)
                or (CONTINUOUS_TEMPERATURE.search(b_clean) and TEMPERATURE_RISE.search(a_clean) and temp_b < temp_a)
            )
            if temp_a != temp_b and (same_claim or sustained_vs_rise):
                same_window = sustained_vs_rise and (
                    temperature_reading_in_window(a_clean, b_clean)
                    if temp_a < temp_b else temperature_reading_in_window(b_clean, a_clean)
                )
                return (
                    NLILabel.CONTRADICTION,
                    ContradictionType.CONFIRMED_CONTRADICTION if same_window else ContradictionType.POSSIBLE_CONTRADICTION,
                    0.95 if same_window else 0.85,
                    f"Temperature discrepancy: Statement A reports {temp_a:g} °C and Statement B reports "
                    f"{temp_b:g} °C, a difference of {abs(temp_a - temp_b):g} °C. "
                    + ("The reported reading falls inside the stated monitoring window on the same date."
                       if same_window else "Check that both statements refer to the same period."),
                    "rules",
                )

        # Check for exact or near-identical agreement
        if a_clean == b_clean or (len(a_clean) > 20 and a_clean in b_clean) or (len(b_clean) > 20 and b_clean in a_clean):
            return (
                NLILabel.ENTAILMENT,
                ContradictionType.AGREEMENT,
                0.95,
                "Both statements assert the identical factual detail across sources.",
                "rules",
            )

        # 1. Attempt NLI Classifier if model pipeline is available
        pipe = self._get_nli_pipeline()
        if pipe is not None:
            try:
                res = pipe({"text": stmt_a, "text_pair": stmt_b}, truncation=True)
                if isinstance(res, list) and len(res) > 0:
                    top_pred = res[0]
                    label_str = top_pred.get("label", "").upper()
                    score = float(top_pred.get("score", 0.0))
                    if "CONTRADICTION" in label_str and score >= 0.70:
                        return (
                            NLILabel.CONTRADICTION,
                            ContradictionType.CONFIRMED_CONTRADICTION,
                            round(score, 2),
                            f"NLI model (RoBERTa-MNLI) identified cross-document contradiction ({round(score * 100, 1)}% confidence).",
                            "nli",
                        )
                    elif "ENTAILMENT" in label_str and score >= 0.75:
                        return (
                            NLILabel.ENTAILMENT,
                            ContradictionType.AGREEMENT,
                            round(score, 2),
                            f"NLI model (RoBERTa-MNLI) identified cross-document agreement ({round(score * 100, 1)}% confidence).",
                            "nli",
                        )
            except Exception as e:
                logger.debug(f"NLI pipeline exception, falling back to rule engine: {e}")

        # 2. Check semantic conflict word pairs (Rule engine)
        for pattern_pos, pattern_neg in CONFLICT_PAIRS:
            match_a_pos = re.search(pattern_pos, a_clean)
            match_b_neg = re.search(pattern_neg, b_clean)
            match_a_neg = re.search(pattern_neg, a_clean)
            match_b_pos = re.search(pattern_pos, b_clean)

            if (match_a_pos and match_b_neg) or (match_a_neg and match_b_pos):
                word1 = match_a_pos.group(0) if match_a_pos else match_a_neg.group(0)
                word2 = match_b_neg.group(0) if match_b_neg else match_b_pos.group(0)
                return (
                    NLILabel.CONTRADICTION,
                    ContradictionType.CONFIRMED_CONTRADICTION,
                    0.92,
                    f"Direct contradiction detected: Statement A states '{word1}' while Statement B asserts '{word2}'.",
                    "rules",
                )

        # 3. Check negation disparity on key verbs (Rule engine)
        verbs = ["entered", "saw", "accessed", "authorized", "present", "left", "downloaded", "swiped", "met", "spoke"]
        for v in verbs:
            in_a = v in a_clean
            in_b = v in b_clean
            if in_a and in_b:
                neg_a = any(neg in a_clean.split() for neg in NEGATION_WORDS)
                neg_b = any(neg in b_clean.split() for neg in NEGATION_WORDS)
                if neg_a != neg_b:
                    return (
                        NLILabel.CONTRADICTION,
                        ContradictionType.CONFIRMED_CONTRADICTION,
                        0.90,
                        f"Opposing assertions regarding action '{v}': one statement affirms it while the other explicitly negates it.",
                        "rules",
                    )

        # A differing clock time is evidence when both passages make the same event claim.
        times_a = list(CLOCK_TIME.finditer(stmt_a))
        times_b = list(CLOCK_TIME.finditer(stmt_b))
        if (len(times_a) == len(times_b) == 1
                and CLOCK_TIME.sub("<time>", a_clean) == CLOCK_TIME.sub("<time>", b_clean)
                and bool(times_a[0][3]) == bool(times_b[0][3])
                and clock_minutes(times_a[0]) != clock_minutes(times_b[0])):
            return (
                NLILabel.CONTRADICTION,
                ContradictionType.POSSIBLE_CONTRADICTION,
                0.85,
                f"Temporal discrepancy: Statement A places the event at {times_a[0][0].strip()} "
                f"and Statement B places it at {times_b[0][0].strip()}.",
                "rules",
            )

        # 5. Check conflicting locations for the same entity at similar time
        loc_indicators = ["server room", "floor 3", "floor 2", "floor 1", "lobby", "outside", "cafeteria", "office", "vault", "home"]
        locs_in_a = [loc for loc in loc_indicators if loc in a_clean]
        locs_in_b = [loc for loc in loc_indicators if loc in b_clean]
        if locs_in_a and locs_in_b and locs_in_a[0] != locs_in_b[0]:
            return (
                NLILabel.CONTRADICTION,
                ContradictionType.CONFIRMED_CONTRADICTION,
                0.88,
                f"Conflicting locations for the same timeframe: Statement A places activity in '{locs_in_a[0]}' while Statement B claims '{locs_in_b[0]}'.",
                "rules",
            )

        # 6. Overlapping affirmation
        words_a = set(re.findall(r"\b[A-Za-z0-9]{3,}\b", a_clean)) - {"the", "and", "that", "this", "with", "from"}
        words_b = set(re.findall(r"\b[A-Za-z0-9]{3,}\b", b_clean)) - {"the", "and", "that", "this", "with", "from"}
        overlap_ratio = len(words_a & words_b) / max(len(words_a | words_b), 1)

        if overlap_ratio > 0.65:
            return (
                NLILabel.ENTAILMENT,
                ContradictionType.AGREEMENT,
                0.85,
                "Substantial factual and lexical overlap confirming the same sequence of events.",
                "rules",
            )

        # 7. Optional LLM Judge for uncertain/borderline cases with moderate lexical overlap
        if overlap_ratio > 0.35:
            data = analysis_model_json(
                'Classify the two claims. Return {"is_contradiction":bool,"is_agreement":bool,'
                '"confidence":number,"explanation":string}. Check whether both claims concern the same event.',
                {"statement_a": stmt_a, "statement_b": stmt_b},
            )
            if data:
                try:
                    confidence = float(data.get("confidence", 0))
                    explanation = data.get("explanation")
                    if not 0.70 <= confidence <= 1 or not isinstance(explanation, str):
                        raise ValueError("Incomplete judge assessment")
                    if data.get("is_contradiction") is True and data.get("is_agreement") is False:
                        return (
                            NLILabel.CONTRADICTION,
                            ContradictionType.POSSIBLE_CONTRADICTION,
                            confidence,
                            explanation,
                            "llm_verified",
                        )
                    elif data.get("is_agreement") is True and data.get("is_contradiction") is False:
                        return (
                            NLILabel.ENTAILMENT,
                            ContradictionType.AGREEMENT,
                            confidence,
                            explanation,
                            "llm_verified",
                        )
                except (TypeError, ValueError):
                    pass

        return (
            NLILabel.NEUTRAL,
            ContradictionType.NEUTRAL,
            0.50,
            "Statements discuss distinct aspects or lack conflicting factual assertions.",
            "rules",
        )

    def find_cross_document_contradictions(
        self,
        case_id: str,
        documents: List[DocumentMetadata],
        entities: List[Entity],
        events: List[Event],
    ) -> List[ContradictionItem]:
        """
        Compares statements across all distinct documents and identifies
        confirmed contradictions, possible contradictions, and key agreements.
        """
        if len(documents) < 2:
            return []

        # Map entities and keywords to sentences across documents
        # Structure: entity_name -> list of (doc, sentence, speaker, timestamp)
        entity_mentions: Dict[str, List[Tuple[DocumentMetadata, str, Optional[str], Optional[str]]]] = {}

        for doc in documents:
            text_val = doc.cleaned_text or doc.raw_text or ""
            sentences = split_into_sentences(text_val)
            if not sentences and text_val.strip():
                sentences = [text_val.strip()]
            for sent in sentences:
                sent = sent.strip()
                if not sent:
                    continue
                # Find which entities are touched
                for ent in entities:
                    if re.search(rf"\b{re.escape(ent.name)}\b", sent, re.IGNORECASE):
                        if ent.name not in entity_mentions:
                            entity_mentions[ent.name] = []
                        entity_mentions[ent.name].append((doc, sent, doc.speaker, doc.timestamp))

        contradictions: List[ContradictionItem] = []
        compared_pairs = set()

        for ent_name, mentions in entity_mentions.items():
            if len(mentions) < 2:
                continue

            for i in range(len(mentions)):
                doc_a, sent_a, spk_a, time_a = mentions[i]
                for j in range(i + 1, len(mentions)):
                    doc_b, sent_b, spk_b, time_b = mentions[j]

                    # MUST be cross-document
                    if doc_a.document_id == doc_b.document_id:
                        continue

                    pair_key = tuple(sorted([sent_a, sent_b]))
                    if pair_key in compared_pairs:
                        continue
                    compared_pairs.add(pair_key)

                    nli_label, c_type, conf, explanation, engine = self.evaluate_pair_full(
                        sent_a, sent_b, entity_context=ent_name
                    )

                    # Only record meaningful conflicts or key corroborated agreements
                    if c_type in [
                        ContradictionType.CONFIRMED_CONTRADICTION,
                        ContradictionType.POSSIBLE_CONTRADICTION,
                        ContradictionType.AGREEMENT
                    ]:
                        auto_confirmed = (
                            c_type == ContradictionType.CONFIRMED_CONTRADICTION
                            and explanation.startswith("Temperature discrepancy:")
                        )
                        c_id = f"contra_{uuid.uuid4().hex[:8]}"
                        contradictions.append(
                            ContradictionItem(
                                id=c_id,
                                case_id=case_id,
                                statement_a=sent_a,
                                statement_b=sent_b,
                                doc_a_id=doc_a.document_id,
                                doc_b_id=doc_b.document_id,
                                doc_a_name=doc_a.filename,
                                doc_b_name=doc_b.filename,
                                speaker_a=spk_a,
                                speaker_b=spk_b,
                                timestamp_a=time_a,
                                timestamp_b=time_b,
                                entity_event_link=ent_name,
                                contradiction_type=c_type,
                                nli_label=nli_label,
                                confidence=conf,
                                detection_engine=engine,
                                explanation=explanation,
                                resolution_status="unresolved" if "contradiction" in c_type.value else "resolved",
                                review_status=(
                                    ReviewStatus.CONFIRMED
                                    if auto_confirmed
                                    else ReviewStatus.PENDING_REVIEW
                                ),
                                reviewer_id="system" if auto_confirmed else None,
                                source_sentences=[sent_a, sent_b],
                            )
                        )

        return contradictions


# Global singleton instance
contradiction_detector = ContradictionDetector()
