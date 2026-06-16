"""Deterministic checks for label/reasoning consistency.

These checks do not try to re-label the case from the report. They only flag
cases where the model's serialized labels appear inconsistent with its own
reasoning or with basic output rules.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Sequence, Set

from Code.config import ALLOWED_LABELS
from Code.utils import normalize_labels
from Code.confidence import min_confidence_percentage


LABEL_ORDER = [
    "NONE",
    "RMCA", "LMCA",
    "RACA", "LACA",
    "RPCA", "LPCA",
    "RPICA", "LPICA",
    "BA",
    "RVA", "LVA",
    "RICA", "LICA",
    "RCA", "LCA",
]

REVIEW_VERSION = "reasoning-label-consistency-v9-bilateral-side-fix"

# Phrases that commonly mean the reasoning rejected a finding/label.
EXCLUSION_PHRASES = (
    "does not qualify",
    "do not qualify",
    "not qualify",
    "non-qualifying",
    "not enough",
    "insufficient",
    "excluded",
    "exclude",
    "omitted",
    "omit",
    "removed",
    "remove",
    "dropped",
    "drop",
    "should not be labeled",
    "should not be included",
    "not labeled",
    "not included",
    "ruled out",
)

# Keep this intentionally narrow to avoid flagging legitimate NONE reasoning that
# mentions a non-qualifying finding, e.g. "right MCA stenosis does not qualify."
POSITIVE_REASONING_PHRASES = (
    "qualifies for",
    "warrants",
    "is labeled",
    "are labeled",
    "should be labeled",
    "therefore label",
    "therefore, label",
    "output",
)

CT_CONTAMINATION_TERMS = (
    "cta",
    "ct angiogram",
    "ct angiography",
    "ctp",
    "ct perfusion",
    "tmax",
    "cbf",
    "cbv",
    "mismatch",
    "hypoperfusion",
    "penumbra",
    "tissue at risk",
    "rapid",
)


# Cases with subdural / extra-axial hemorrhage are high-risk for false
# ischemic territory labels.  In these cases, subtle adjacent hypodensity may
# represent hemorrhage/trauma-related edema or contusion rather than an acute
# arterial infarct.  These terms trigger manual review; they do not change the
# model's labels by themselves.
SUBDURAL_EXTRA_AXIAL_REVIEW_TERMS = (
    "subdural hematoma",
    "subdural haemorrhage",
    "subdural hemorrhage",
    "subdural blood",
    "subdural collection",
    "acute subdural",
    "sdh",
    "extra-axial hematoma",
    "extra-axial haemorrhage",
    "extra-axial hemorrhage",
    "extra-axial blood",
    "extraaxial hematoma",
    "extraaxial hemorrhage",
)

SUBARACHNOID_REVIEW_TERMS = (
    "subarachnoid hemorrhage",
    "subarachnoid haemorrhage",
    "subarachnoid blood",
    "sah",
)

SUBDURAL_SCAN_FIELDS = (
    ("CT_Report", "CT report"),
    ("New_CT_Report", "sanitized CT report"),
    ("MRI_Report", "MRI report"),
    ("CTA_Report", "CTA brain-window text"),
    ("CTP_Report", "CTP context text"),
)

LABEL_VARIANTS = {
    "RMCA": ["rmca", "right mca", "right middle cerebral", "right m1", "right m2", "right m3", "right m4"],
    "LMCA": ["lmca", "left mca", "left middle cerebral", "left m1", "left m2", "left m3", "left m4"],
    "RACA": ["raca", "right aca", "right anterior cerebral", "right a1", "right a2", "right a3"],
    "LACA": ["laca", "left aca", "left anterior cerebral", "left a1", "left a2", "left a3"],
    "RPCA": ["rpca", "right pca", "right posterior cerebral", "right p1", "right p2", "right p3", "right p4"],
    "LPCA": ["lpca", "left pca", "left posterior cerebral", "left p1", "left p2", "left p3", "left p4"],
    "RPICA": ["rpica", "right pica", "right posterior inferior cerebellar", "right inferior cerebellar"],
    "LPICA": ["lpica", "left pica", "left posterior inferior cerebellar", "left inferior cerebellar"],
    "BA": ["ba", "basilar", "basilar artery", "basilar tip", "basilar trunk", "pons", "pontine", "central pons", "paramedian pons"],
    "RVA": ["rva", "right vertebral", "right vertebral artery", "right v1", "right v2", "right v3", "right v4", "right intradural vertebral"],
    "LVA": ["lva", "left vertebral", "left vertebral artery", "left v1", "left v2", "left v3", "left v4", "left intradural vertebral"],
    "RICA": ["rica", "right ica", "right internal carotid", "right carotid terminus", "right intracranial carotid"],
    "LICA": ["lica", "left ica", "left internal carotid", "left carotid terminus", "left intracranial carotid"],
    "RCA": ["rca", "right common carotid", "right cca", "right cervical carotid"],
    "LCA": ["lca", "left common carotid", "left cca", "left cervical carotid"],
    "NONE": ["none", "no qualifying", "no evidence", "negative"],
}

TERRITORY_TERMS = {
    "MCA": [
        "mca", "middle cerebral", "m1", "m2", "m3", "m4",
        "insula", "insular", "operculum", "basal ganglia", "lentiform",
        "putamen", "caudate", "internal capsule", "corona radiata",
        "centrum semiovale",
    ],
    "ACA": [
        "aca", "anterior cerebral", "a1", "a2", "a3", "pericallosal",
        "callosomarginal", "medial frontal", "medial parietal", "parafalcine",
        "cingulate", "corpus callosum",
    ],
    "PCA": [
        "pca", "posterior cerebral", "p1", "p2", "p3", "p4",
        "occipital", "calcarine", "posterior temporal", "thalamus", "thalamic",
    ],
    "PICA": [
        "pica", "posterior inferior cerebellar", "inferior cerebellar",
        "cerebellar hemisphere", "cerebellum",
    ],
    "BA": [
        "ba", "basilar", "basilar artery", "basilar tip", "basilar trunk",
        "pons", "pontine", "central pons", "paramedian pons", "brainstem",
    ],
    "VA": [
        "vertebral", "vertebral artery", "v1", "v2", "v3", "v4",
        "intradural vertebral", "vertebrobasilar junction",
    ],
    "ICA": [
        "ica", "internal carotid", "carotid terminus", "intracranial carotid",
        "petrous", "cavernous", "paraclinoid", "supraclinoid",
    ],
    "CA": ["common carotid", "cca", "cervical carotid"],
}

LABEL_SIDE_TERRITORY = {
    "RMCA": ("right", "MCA"),
    "LMCA": ("left", "MCA"),
    "RACA": ("right", "ACA"),
    "LACA": ("left", "ACA"),
    "RPCA": ("right", "PCA"),
    "LPCA": ("left", "PCA"),
    "RPICA": ("right", "PICA"),
    "LPICA": ("left", "PICA"),
    "RVA": ("right", "VA"),
    "LVA": ("left", "VA"),
    "RICA": ("right", "ICA"),
    "LICA": ("left", "ICA"),
    "RCA": ("right", "CA"),
    "LCA": ("left", "CA"),
}

FIELD_REASONING_PAIRS = [
    ("CT_Original_GT", "CT_Original_GT_reasoning", "CT original"),
    ("CT_GT", "CT_GT_reasoning", "CT"),
    ("CTA_GT", "CTA_GT_reasoning", "CTA"),
    ("CTP_GT", "CTP_GT_reasoning", "CTP"),
    ("Combined_GT", "CT_Combined_GT_reasoning", "Combined"),
]

CONFIDENCE_FIELDS = [
    ("CT_Original_GT", "CT original"),
    ("CT_GT", "CT"),
    ("CTA_GT", "CTA"),
    ("CTP_GT", "CTP"),
    ("Combined_GT", "Combined"),
]


def _compact_text(value: Any) -> str:
    text = str(value or "").lower()
    text = text.replace("/", " ")
    return re.sub(r"\s+", " ", text).strip()


def _reasoning_for_consistency(value: Any) -> str:
    """Return only the final/winning reasoning section for consistency checks.

    Confidence mode can store a union of all sample reasonings in the same
    reasoning field, including alternate-label samples. The deterministic
    label-vs-reasoning checks should not treat alternate votes as if they were
    support for the final label, so this strips the alternate section before
    checking consistency.
    """
    text = str(value or "")
    marker = "\n\nAlternate-label reasoning samples:"
    if marker in text:
        return text.split(marker, 1)[0]
    return text


def _sort_labels(labels: Iterable[str]) -> List[str]:
    order = {label: idx for idx, label in enumerate(LABEL_ORDER)}
    unique = []
    for label in labels:
        label = str(label).strip().upper()
        if label not in unique:
            unique.append(label)
    return sorted(unique, key=lambda label: order.get(label, 999))


def _label_variants(label: str) -> List[str]:
    return LABEL_VARIANTS.get(label, [label.lower()])


def _variant_regex(variant: str) -> str:
    # Keep word boundaries but allow flexible spaces inside phrases.
    parts = [re.escape(part) for part in variant.split()]
    return r"\b" + r"\s+".join(parts) + r"\b"


def _contains_variant(text: str, label: str) -> bool:
    return any(re.search(_variant_regex(variant), text) for variant in _label_variants(label))


def _sentences(text: str) -> List[str]:
    """Split reasoning into rough sentences/clauses for safer local checks."""
    compact = _compact_text(text)
    return [part.strip() for part in re.split(r"(?<=[.!?;])\s+", compact) if part.strip()]


def _labels_in_text(text: str) -> Set[str]:
    found: Set[str] = set()
    for label in LABEL_ORDER:
        if label == "NONE":
            continue
        if _contains_variant(text, label):
            found.add(label)
    return found


def _remove_excluded_tail(sentence: str) -> str:
    """Ignore text that lists rejected/explanatory labels after a positive only-phrase.

    Example:
    "only RMCA is labeled because RICA is not included"
    should count RMCA as supported, but should not count RICA as supported.
    """
    cut_words = [
        " omitting ", " omits ", " omit ",
        " excluding ", " excludes ", " exclude ",
        " rather than ", " instead of ", " without ",
        " because ", " since ", " so ", " therefore ",
        " and no ", " but no ", " however no ",
        " and not ", " but not ", " however not ",
    ]
    cut_positions = [sentence.find(word) for word in cut_words if sentence.find(word) != -1]
    if not cut_positions:
        return sentence
    return sentence[: min(cut_positions)]


def _labels_after_only_phrase(reasoning: str) -> Set[str]:
    """Return labels from explicit 'only X is labeled/included/output' sentences.

    This intentionally ignores incidental words like "stenosis-only" and ignores
    excluded labels that appear later in the same sentence, e.g.
    "only LMCA and LACA are output, omitting LICA".
    """
    found: Set[str] = set()
    for sentence in _sentences(reasoning):
        if "only" not in sentence:
            continue
        if not re.search(r"\bonly\b", sentence):
            continue
        # Only treat it as a supported-label sentence if it uses final-label verbs.
        if not re.search(r"\b(labeled|labelled|included|output|returned|assigned)\b", sentence):
            continue
        # Only consider labels after the word "only"; labels in earlier context
        # such as "ICA thrombus plus MCA/ACA" are background, not the supported final labels.
        support_part = sentence[re.search(r"\bonly\b", sentence).start():]
        support_part = _remove_excluded_tail(support_part)
        labels_here = _labels_in_text(support_part)
        if labels_here:
            found.update(labels_here)
    return found

def _label_excluded_by_reasoning(reasoning: str, label: str) -> bool:
    """Return True only when exclusion wording directly targets this label.

    This intentionally avoids broad proximity checks.  Sentences like
    "only LMCA and LACA are output, omitting LICA" should not flag LMCA/LACA;
    only a direct phrase such as "omit LMCA" or "LMCA is omitted" should flag
    LMCA.  This keeps wording-only review false positives low.
    """
    for sentence in _sentences(reasoning):
        if not _contains_variant(sentence, label):
            continue

        # Ignore common CT-contamination phrasing where the CTA/CTP source is
        # excluded but the same territory is supported by CT-visible findings.
        if ("cta" in sentence or "ct angiogram" in sentence or "ct perfusion" in sentence) and re.search(
            r"\b(but|however)\b[^.]*\b(sufficient|supports|support|label|labeled|labelled|warrants)\b",
            sentence,
        ):
            continue

        # Avoid false positives where the sentence excludes an upstream ICA label
        # while keeping a downstream non-ICA label.
        if label not in {"RICA", "LICA"} and re.search(
            r"\b(ica|internal carotid|carotid)\b[^.;]{0,80}\b(not labeled|not labelled|not included|omitted|excluded|removed|dropped)\b",
            sentence,
        ):
            if any(term in sentence for term in ["downstream mca", "mca territory", "specific downstream", "downstream territory"]):
                continue

        for variant in _label_variants(label):
            v = _variant_regex(variant)
            label_ref = rf"(?:the\s+)?{v}(?:\s+(?:label|territory|finding))?"
            label_ref_after_verb = rf"(?:the\s+)?(?:label\s+)?{v}(?:\s+(?:label|territory|finding))?"

            # Direct-target exclusions only.  Do NOT use broad "label within 70
            # chars of omitted" patterns; those caused false flags when the
            # sentence listed included labels and then omitted a different label.
            direct_exclusion_patterns = [
                # "LMCA does not qualify" / "left mca does not qualify"
                rf"{label_ref}\s+(?:does|do|did)\s+not\s+qualify\b",
                rf"{label_ref}\s+(?:is|are|was|were)\s+non[-\s]?qualifying\b",
                # "LMCA is not included/labeled/output/assigned"
                rf"{label_ref}\s+(?:is|are|was|were)\s+not\s+(?:included|labeled|labelled|output|returned|assigned)\b",
                # "LMCA is omitted/excluded/removed/dropped"
                rf"{label_ref}\s+(?:is|are|was|were)\s+(?:omitted|excluded|removed|dropped)\b",
                # "LMCA should be omitted" / "LMCA should not be labeled"
                rf"{label_ref}\s+(?:should|must|would|can)\s+be\s+(?:omitted|excluded|removed|dropped)\b",
                rf"{label_ref}\s+(?:should|must|would|can)\s+not\s+be\s+(?:included|labeled|labelled|output|returned|assigned)\b",
                # "not enough evidence for LMCA" / "insufficient evidence for LMCA"
                rf"\b(?:not\s+enough|insufficient)\s+(?:evidence\s+)?(?:for|to\s+support)\s+{label_ref_after_verb}\b",
                # "omit LMCA" / "excluding the right MCA label" / "do not include LMCA"
                rf"\b(?:omit|omits|omitting|omitted|exclude|excludes|excluding|excluded|remove|removes|removing|removed|drop|drops|dropping|dropped)\s+{label_ref_after_verb}\b",
                rf"\b(?:do|does|did)\s+not\s+(?:include|label|output|return|assign)\s+{label_ref_after_verb}\b",
            ]
            if any(re.search(pattern, sentence) for pattern in direct_exclusion_patterns):
                return True

            # Explicit retention should suppress broad/abstract exclusion wording
            # elsewhere in the same sentence, but only after direct exclusions
            # above have been checked.  This avoids treating "do not include
            # RPCA" as positive support just because it contains "include RPCA".
            positive_retention_patterns = [
                rf"{label_ref}\s+(?:must|should)\s+be\s+(?:retained|kept|included|labeled|labelled)\b",
                rf"{label_ref}\s+(?:is|are|was|were)\s+(?:retained|kept|included|labeled|labelled)\b",
                rf"\b(?:retain|retains|retained|keeping|keep|keeps|kept|include|includes|included)\s+{label_ref_after_verb}\b",
            ]
            if any(re.search(pattern, sentence) for pattern in positive_retention_patterns):
                continue
    return False

def _has_positive_reasoning_for_any_territory(reasoning: str) -> bool:
    text = _compact_text(reasoning)
    has_positive_phrase = any(phrase in text for phrase in POSITIVE_REASONING_PHRASES)
    if not has_positive_phrase:
        return False

    # Avoid false positives from sentences like "RMCA does not qualify".
    for label in LABEL_ORDER:
        if label == "NONE":
            continue
        if _contains_variant(text, label) and not _label_excluded_by_reasoning(reasoning, label):
            return True
    return False


def _has_side_territory(text: str, side: str, territory: str) -> bool:
    """Detect whether side and territory terms appear close together."""
    side_words = [side]
    if side == "right":
        side_words += ["rt"]
    elif side == "left":
        side_words += ["lt"]

    terms = TERRITORY_TERMS.get(territory, [])
    for side_word in side_words:
        side_re = re.escape(side_word)
        for term in terms:
            term_re = _variant_regex(term)
            patterns = [
                rf"\b{side_re}\b(?:\W+\w+){{0,5}}\W+{term_re}",
                rf"{term_re}(?:\W+\w+){{0,5}}\W+\b{side_re}\b",
            ]
            if any(re.search(pattern, text) for pattern in patterns):
                return True
    return False


def _opposite_side_label(label: str) -> str | None:
    """Return the same-territory label on the opposite side, if available."""
    if label not in LABEL_SIDE_TERRITORY:
        return None

    side, territory = LABEL_SIDE_TERRITORY[label]
    opposite_side = "left" if side == "right" else "right"
    for candidate, (candidate_side, candidate_territory) in LABEL_SIDE_TERRITORY.items():
        if candidate_side == opposite_side and candidate_territory == territory:
            return candidate
    return None


def _has_bilateral_territory_context(text: str, territory: str) -> bool:
    """Return True when reasoning clearly describes bilateral/both-sided territory involvement.

    This prevents false side-mismatch flags for labels such as RMCA + LMCA
    when the reasoning says "bilateral MCA infarcts" or "both MCA territories"
    without repeating separate right-MCA and left-MCA phrases.
    """
    territory_terms = TERRITORY_TERMS.get(territory, [])
    if not territory_terms:
        return False

    has_territory = any(re.search(_variant_regex(term), text) for term in territory_terms)
    if not has_territory:
        return False

    bilateral_patterns = (
        r"\bbilateral(?:ly)?\b",
        r"\bbi[-\s]?hemispheric\b",
        r"\bboth\s+(?:sides|hemispheres|territories|mca|aca|pca|pica|vertebral|ica)\b",
        r"\b(?:right|left)\s+(?:and|/)\s+(?:left|right)\b",
        r"\b(?:left|right)\s*(?:>|greater\s+than|more\s+than)\s*(?:right|left)\b",
        r"\b(?:right|left)\s+greater\s+than\s+(?:left|right)\b",
    )
    return any(re.search(pattern, text) for pattern in bilateral_patterns)


def _side_mismatch_flags(field_name: str, labels: Sequence[str], reasoning: str) -> List[str]:
    text = _compact_text(reasoning)
    flags: List[str] = []
    label_set = set(labels)

    for label in labels:
        if label not in LABEL_SIDE_TERRITORY:
            continue
        side, territory = LABEL_SIDE_TERRITORY[label]
        opposite = "left" if side == "right" else "right"
        has_expected = _has_side_territory(text, side, territory) or _contains_variant(text, label)
        has_opposite = _has_side_territory(text, opposite, territory)

        opposite_label = _opposite_side_label(label)

        # If both sides of the same territory are already present in the final
        # label set, bilateral wording can legitimately support both labels
        # without spelling out each side separately.  Do not flag LMCA just
        # because the same sentence says "right greater than left" or cites
        # right-sided dominance in a bilateral MCA case.
        if (
            opposite_label in label_set
            and _has_bilateral_territory_context(text, territory)
        ):
            continue

        if has_opposite and not has_expected:
            flags.append(
                f"{field_name}: label {label} is {side} {territory}, but reasoning appears to cite {opposite} {territory} evidence"
                + (f" ({opposite_label})" if opposite_label else "")
            )
    return flags


def _common_carotid_flags(field_name: str, labels: Sequence[str], reasoning: str) -> List[str]:
    """Flag RCA/LCA when reasoning appears to describe intracranial territories instead."""
    text = _compact_text(reasoning)
    flags: List[str] = []
    for label in labels:
        if label not in {"RCA", "LCA"}:
            continue
        side, _ = LABEL_SIDE_TERRITORY[label]
        has_common_carotid = _has_side_territory(text, side, "CA") or _contains_variant(text, label)
        mentions_intracranial = any(
            _has_side_territory(text, side, territory)
            for territory in ["MCA", "ACA", "PCA", "PICA", "VA", "ICA"]
        )
        if mentions_intracranial and not has_common_carotid:
            flags.append(
                f"{field_name}: label {label} is common/cervical carotid, but reasoning appears to cite intracranial territory evidence"
            )
    return flags


def check_field_consistency(field_name: str, labels_raw: Any, reasoning_raw: Any) -> List[str]:
    """Return review flags for one label field and its reasoning."""
    labels = normalize_labels(labels_raw)
    reasoning = _reasoning_for_consistency(reasoning_raw)
    text = _compact_text(reasoning)
    flags: List[str] = []

    invalid = []
    if isinstance(labels_raw, list):
        invalid = [str(label).strip().upper() for label in labels_raw if str(label).strip().upper() not in ALLOWED_LABELS]
    elif labels_raw is not None:
        invalid = [str(labels_raw)]
    if invalid:
        flags.append(f"{field_name}: invalid label(s) before normalization: {invalid}")

    if "NONE" in labels and len(labels) > 1:
        flags.append(f"{field_name}: NONE appears with territory labels")

    # Be conservative with NONE: many correct NONE explanations mention a
    # non-qualifying vessel/territory before rejecting it. Do not flag NONE
    # solely from positive-looking words unless you add a stricter project rule.

    only_labels = _labels_after_only_phrase(reasoning)
    non_none_labels = set(label for label in labels if label != "NONE")
    if only_labels and non_none_labels and only_labels != non_none_labels:
        flags.append(
            f"{field_name}: reasoning uses an 'only' phrase supporting {_sort_labels(only_labels)}, but labels are {_sort_labels(non_none_labels)}"
        )
    for label in non_none_labels:
        if _label_excluded_by_reasoning(reasoning, label):
            flags.append(f"{field_name}: reasoning appears to exclude/omit label {label}, but it is present in labels")

    flags.extend(_side_mismatch_flags(field_name, labels, reasoning))
    flags.extend(_common_carotid_flags(field_name, labels, reasoning))

    return flags


def _contamination_scan_text(report: str) -> str:
    """Scan clinically meaningful CT body text, not administrative COMPARISON text."""
    text = _compact_text(report)
    # Remove the comparison section when present because it often names CTA/CTP
    # studies without importing their findings.
    text = re.sub(r"\bcomparison\b:?.*?(?=\btechnique\b|\bfindings\b|\bimpression\b|$)", " ", text)
    return _compact_text(text)


def check_sanitized_ct(case: Dict[str, Any]) -> List[str]:
    flags: List[str] = []
    if not bool(case.get("CT_Report_Was_Sanitized", False)):
        return flags

    original = normalize_labels(case.get("CT_Original_GT"))
    final = normalize_labels(case.get("CT_GT"))
    if set(original) != set(final):
        flags.append(f"CT sanitization changed CT labels from {_sort_labels(original)} to {_sort_labels(final)}; review the sanitized CT report")

    new_report = _contamination_scan_text(case.get("New_CT_Report", ""))
    remaining_terms = [term for term in CT_CONTAMINATION_TERMS if re.search(_variant_regex(term), new_report)]
    if remaining_terms:
        flags.append(f"New_CT_Report still contains possible CTA/CTP contamination terms outside COMPARISON: {remaining_terms[:5]}")

    return flags


def check_ctp_special_cases(case: Dict[str, Any]) -> List[str]:
    """Catch high-risk CTP wording problems that ordinary consistency checks miss."""
    flags: List[str] = []
    ctp_labels = normalize_labels(case.get("CTP_GT"))
    ctp_reasoning = _compact_text(case.get("CTP_GT_reasoning", ""))

    if any(label in ctp_labels for label in ["RPICA", "LPICA"]):
        # Project rule: malformed/shorthand wording such as "left-sided cord infarct"
        # can legitimately map to LPICA when the CTP report gives no better named
        # anterior-circulation territory. Do not flag that rule-consistent output.
        pica_terms = ["pica", "posterior inferior cerebellar", "inferior cerebellar", "cerebellar", "posterior fossa", "posterior-fossa", "cord infarct"]
        if not any(term in ctp_reasoning for term in pica_terms):
            flags.append(
                "CTP: PICA label is present, but reasoning does not cite PICA/cerebellar/posterior-fossa/cord-infarct localization"
            )

    return flags



def _sentence_has_negated_term(sentence: str, term: str) -> bool:
    """Return True when a hemorrhage term appears in a clearly negative phrase."""
    term_re = _variant_regex(term)
    negation_before = re.search(
        rf"\b(no|without|negative for|absence of|no evidence of|no findings of|not demonstrating|not showing)\b[^.;]{{0,80}}{term_re}",
        sentence,
    )
    negation_after = re.search(
        rf"{term_re}[^.;]{{0,50}}\b(absent|not seen|not present|not identified|not demonstrated|ruled out)\b",
        sentence,
    )
    return bool(negation_before or negation_after)


def _first_positive_hemorrhage_sentence(report: Any, terms: Sequence[str]) -> str:
    """Return the first sentence containing a non-negated review term."""
    for sentence in _sentences(str(report or "")):
        for term in terms:
            if re.search(_variant_regex(term), sentence) and not _sentence_has_negated_term(sentence, term):
                return sentence
    return ""


def _short_quote(text: str, max_chars: int = 180) -> str:
    text = re.sub(r"\s+", " ", str(text or "")).strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def check_subdural_extra_axial_context(case: Dict[str, Any]) -> List[str]:
    """Flag reports with subdural/extra-axial hemorrhage context for manual review.

    This intentionally does not decide whether the label is wrong. It simply
    marks cases where extra-axial hemorrhage can make a nearby subtle
    hypodensity look like an arterial territory infarct.
    """
    flags: List[str] = []
    seen_sentences: Set[str] = set()

    for field_name, display_name in SUBDURAL_SCAN_FIELDS:
        report = case.get(field_name, "")
        if not str(report or "").strip():
            continue

        subdural_sentence = _first_positive_hemorrhage_sentence(report, SUBDURAL_EXTRA_AXIAL_REVIEW_TERMS)
        if subdural_sentence and subdural_sentence not in seen_sentences:
            seen_sentences.add(subdural_sentence)
            flags.append(
                "Subdural/extra-axial hemorrhage context detected in "
                f"{display_name}; review any subtle hypodensity or territory label because extra-axial blood is not an ischemic stroke territory. "
                f"Evidence: {_short_quote(subdural_sentence)}"
            )

        # Also flag SAH when it appears with a subtle hypodensity, because this
        # was the failure pattern in case 108867337_20250824_17223200.
        sah_sentence = _first_positive_hemorrhage_sentence(report, SUBARACHNOID_REVIEW_TERMS)
        report_text = _compact_text(report)
        has_subtle_hypodensity = "subtle hypodensity" in report_text or "questionable hypodensity" in report_text
        if sah_sentence and has_subtle_hypodensity and sah_sentence not in seen_sentences:
            seen_sentences.add(sah_sentence)
            flags.append(
                "Subarachnoid hemorrhage plus subtle/questionable hypodensity detected; review before assigning a PCA/MCA/ACA ischemic territory label. "
                f"Evidence: {_short_quote(sah_sentence)}"
            )

    return flags

def check_confidence_fields(case: Dict[str, Any]) -> List[str]:
    """Flag label fields whose repeated-run vote share is below threshold."""
    flags: List[str] = []
    threshold = min_confidence_percentage()

    for prefix, display_name in CONFIDENCE_FIELDS:
        is_confident_key = f"{prefix}_is_confident"
        pct_key = f"{prefix}_confidence_percentage"
        answers_key = f"{prefix}_possible_answers"

        # Only run this check when confidence fields are present.
        if is_confident_key not in case and pct_key not in case:
            continue

        try:
            pct = float(case.get(pct_key, 0) or 0)
        except (TypeError, ValueError):
            pct = 0.0

        if not bool(case.get(is_confident_key, False)) or pct < threshold:
            flags.append(
                f"{display_name}: low confidence vote share ({pct:.2f}% < {threshold:.2f}%); "
                f"possible answers: {case.get(answers_key, [])}"
            )

    return flags


def build_review_flags(case: Dict[str, Any]) -> List[str]:
    flags: List[str] = []
    for label_field, reasoning_field, display_name in FIELD_REASONING_PAIRS:
        if label_field not in case:
            continue
        flags.extend(check_field_consistency(display_name, case.get(label_field), case.get(reasoning_field, "")))
    flags.extend(check_sanitized_ct(case))
    flags.extend(check_ctp_special_cases(case))
    flags.extend(check_subdural_extra_axial_context(case))
    flags.extend(check_confidence_fields(case))

    # Deduplicate while preserving order.
    deduped: List[str] = []
    for flag in flags:
        if flag not in deduped:
            deduped.append(flag)
    return deduped



def _case_has_positive_territory_label(case: Dict[str, Any]) -> bool:
    """Return True if any output label field has a non-NONE territory label."""
    for field in ("CT_Original_GT", "CT_GT", "CTA_GT", "CTP_GT", "Combined_GT"):
        labels = normalize_labels(case.get(field, ["NONE"]))
        if labels and labels != ["NONE"]:
            return True
    return False


def _review_flag_priority(case: Dict[str, Any], flag: str) -> str:
    """Classify review flags into red/high-priority or yellow/warning.

    Red flags are likely to affect correctness or stability. Yellow flags are
    kept visible but are often wording-only/noisy checks.
    """
    text = _compact_text(flag)

    if any(term in text for term in ("subdural", "extra axial", "extra-axial", "subarachnoid", "sah")):
        return "red" if _case_has_positive_territory_label(case) else "yellow"

    red_terms = (
        "low confidence vote share",
        "invalid label",
        "none appears with territory labels",
        "ct sanitization changed ct labels",
        "new_ct_report still contains possible cta ctp contamination terms",
        "pica label is present",
        "but reasoning appears to cite",
        "common/cervical carotid",
    )
    if any(term in text for term in red_terms):
        return "red"

    # Reasoning wording checks are intentionally warning-only. They are useful
    # for spotting contradictions, but they are the most common false-positive
    # source and should not make the case a high-priority review by themselves.
    yellow_terms = (
        "reasoning appears to exclude/omit label",
        "reasoning uses an 'only' phrase",
    )
    if any(term in text for term in yellow_terms):
        return "yellow"

    return "yellow"


def split_review_flags_by_priority(case: Dict[str, Any], flags: List[str]) -> Dict[str, List[str]]:
    """Return review flags split into red/yellow priority buckets."""
    red: List[str] = []
    yellow: List[str] = []
    for flag in flags:
        bucket = red if _review_flag_priority(case, flag) == "red" else yellow
        if flag not in bucket:
            bucket.append(flag)
    return {"red": red, "yellow": yellow}


def add_review_flags(case: Dict[str, Any]) -> Dict[str, Any]:
    """Return a copy of a case with red review flags and yellow warnings.

    Needs_Review is now reserved for red/high-priority flags.  Yellow wording
    warnings are still preserved in Review_Flags and Has_Warnings, but they do
    not make the case a high-priority review by themselves.
    """
    updated = dict(case)
    flags = build_review_flags(updated)
    split_flags = split_review_flags_by_priority(updated, flags)
    red_flags = split_flags["red"]
    yellow_flags = split_flags["yellow"]

    updated["Needs_Review"] = bool(red_flags)
    updated["Has_Warnings"] = bool(yellow_flags)
    updated["Review_Flags_Red"] = red_flags
    updated["Review_Flags_Yellow"] = yellow_flags
    updated["Review_Flags"] = red_flags + yellow_flags
    updated["Review_Flag_Count"] = len(red_flags) + len(yellow_flags)
    updated["Review_Flag_Red_Count"] = len(red_flags)
    updated["Review_Flag_Yellow_Count"] = len(yellow_flags)
    updated["Review_Check_Version"] = REVIEW_VERSION
    return updated
