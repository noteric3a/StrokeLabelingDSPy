import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List

import config as cfg
from config import ALLOWED_LABELS


LABEL_ORDER = [
    "NONE",
    "RMCA", "LMCA",
    "RACA", "LACA",
    "RPCA", "LPCA",
    "RPICA", "LPICA",
    "BA",
    "RVA", "LVA",
    "RICA", "LICA",
]


def labels_text() -> str:
    return ", ".join(ALLOWED_LABELS)


def label_order_text() -> str:
    return ", ".join([label for label in LABEL_ORDER if label in ALLOWED_LABELS])


def final_consistency_check(label_field: str = "labels") -> str:
    return f"""
Final consistency check before returning JSON:
- Re-read the reasoning you are about to output.
- Identify the exact labels that your reasoning supports.
- Set {label_field} to exactly those labels and no others.
- If your reasoning says "only X", {label_field} must contain only X.
- If your reasoning says a label is excluded, omitted, does not qualify, should be removed, or should not be labeled, that label must not appear in {label_field}.
- If two label sets seem possible, choose the smaller label set that is directly supported by explicit report wording; do not add extra labels from mechanism, broad vascular supply, or uncertainty.
- If there is any conflict between your reasoning and {label_field}, revise {label_field} before output.
""".strip()


def base_rules() -> str:
    return f"""
Allowed labels:
{labels_text()}

Output rules:
- Return JSON only. No markdown, comments, or extra text.
- Every label field must contain at least one label.
- Use ["NONE"] only when the report gives no qualifying acute/recent/current stroke-territory evidence.
- If any territory label is present, do not include "NONE".
- If multiple labels are needed, use this order: {label_order_text()}.
- Reasoning must only cite evidence from the report being labeled.

Labeling policy:
- Score acute stroke territory, not every old/subacute/chronic infarct ever seen.
- Clear subacute-only, evolving-known, remote, old, chronic, postoperative, moyamoya/bypass-related, or comparison-follow-up infarcts are negative unless the same report independently gives a qualifying acute finding.
- Wording such as "acute/subacute" or "acute to subacute" can qualify only when it is presented as the current suspected acute event and is not contradicted by "no acute findings", "subacute only", chronic disease, or a prior/evolving follow-up context.

Deterministic tie-breaker policy for repeated/confidence runs:
- Prefer the most literal named side + named vessel/territory in the report.
- Do not expand a label set because a finding could theoretically affect another vascular territory; add a label only when that territory has its own explicit evidence.
- When the report wording is borderline, choose the smaller/conservative label set rather than adding uncertain secondary territories.
- If the only disagreement is whether to add an upstream ICA/carotid label to an already localized CT/CTP tissue/perfusion territory, do not add ICA unless the report itself frames the acute abnormality as an unseparated ICA/carotid-territory pattern.
- Never change side based on mechanism or comparison text; right-sided evidence maps to right labels and left-sided evidence maps to left labels.

Core territory map:
- MCA: MCA, M1, M2, M3, M4, insula, operculum, frontal operculum, Sylvian fissure, basal ganglia, lentiform nucleus, putamen, caudate, internal capsule, corona radiata, centrum semiovale, lateral frontal/parietal/temporal cortex, frontoparietal convexity.
- ACA: ACA, A1, A2, A3, pericallosal, callosomarginal, medial frontal/parietal, parafalcine region, cingulate gyrus, corpus callosum, supplementary motor area.
- PCA: PCA, P1, P2, P3, P4, occipital lobe, calcarine cortex, posterior temporal lobe, thalamus.
- PICA: PICA, posterior inferior cerebellum, inferior cerebellar hemisphere.
- BA: basilar artery, BA, basilar tip, basilar trunk, pons, central pons, paramedian pons, pontine infarct, brainstem infarct when described as basilar-territory.
- VA: vertebral artery, vertebral arteries, V1, V2, V3, V4, intradural vertebral artery, vertebrobasilar junction.
- ICA: ICA, internal carotid artery, petrous/cavernous/paraclinoid/supraclinoid/intracranial ICA, ICA terminus, carotid terminus.

Basilar / vertebral rule:
- BA is an available label when ALLOWED_LABELS includes "BA".
- RVA/LVA are available labels when ALLOWED_LABELS includes "RVA" and "LVA".
- Definite acute basilar artery occlusion, thrombus, flow cutoff, filling defect, absent opacification, or near-occlusion maps to BA.
- Definite acute right vertebral artery occlusion, thrombus, flow cutoff, filling defect, absent opacification, nonopacification, or near-occlusion maps to RVA.
- Definite acute left vertebral artery occlusion, thrombus, flow cutoff, filling defect, absent opacification, nonopacification, or near-occlusion maps to LVA.
- Acute/recent infarct, ischemia, restricted diffusion, hypodensity, infarct core, or hypoperfusion in the pons/central pons/paramedian pons/brainstem when described as basilar-territory maps to BA.
- Do not force basilar or vertebral artery disease into PCA or PICA. Add PCA/PICA only when the report separately gives definite PCA/PICA branch involvement or mapped acute infarct/perfusion territory.
- Basilar artery occlusion plus a definite right P1/PCA occlusion should be BA + RPCA; basilar artery alone should be BA.
- Vertebral artery occlusion alone should be RVA/LVA, not BA or PICA, unless the report separately identifies basilar involvement, PICA involvement, or an inferior cerebellar/PICA-territory infarct/perfusion abnormality.

Hard mapping rules:
- MCA-named findings map to RMCA/LMCA, not ACA.
- Insula, operculum, basal ganglia, lentiform nucleus, putamen, caudate, internal capsule, corona radiata, and centrum semiovale map to MCA unless the report explicitly says another vascular territory.
- Corona radiata and centrum semiovale map to MCA even when described as frontal, posterior frontal, frontoparietal, small, questionable, or lacunar. Do not label them ACA unless the report explicitly says ACA territory/A1/A2/A3/pericallosal/callosomarginal/medial/parafalcine/cingulate/corpus callosum.
- Do not map frontal/frontoparietal/parietal wording to ACA unless the report says ACA/A1/A2/A3, medial, parafalcine, cingulate, corpus callosum, pericallosal, or callosomarginal.
- Occipital, calcarine, posterior temporal, and thalamic territorial findings map to PCA, not ACA/MCA.
- ICA/carotid terminus is an upstream/general vessel label, not a parenchymal tissue label.
- For CT and CTP labeling, prefer the specific downstream tissue/perfusion territory (MCA or ACA) when the report localizes the abnormality there.
- Use RICA/LICA for CT or CTP only when the report explicitly frames the acute abnormality as an ICA/carotid-terminus/carotid-territory pattern and does not provide a more specific MCA/ACA territory.
- Do not add RICA/LICA just because an upstream ICA/carotid thrombus could explain downstream MCA/ACA findings.
- Do not output RICA/LICA for infarcts in basal ganglia, corona radiata, centrum semiovale, internal capsule, insula, or lacunar locations; those are MCA unless the report explicitly names another territory.
- Common-carotid/cervical-carotid labels are not available unless ALLOWED_LABELS explicitly includes them.

Evidence rules:
- Positive evidence includes acute/recent infarct, acute/recent ischemia, new infarct, restricted diffusion, acute hypodensity, loss of gray-white differentiation, definite dense vessel sign, definite thrombus, occlusion, flow cutoff, filling defect, near-occlusion, hypoperfusion, mismatch, penumbra, tissue at risk, brain at risk, or infarct core.
- A small acute/recent lacunar infarct still counts if it is in a mapped acute territory such as corona radiata, centrum semiovale, internal capsule, basal ganglia, or putamen; map those to MCA unless the report explicitly names ACA/PCA/PICA.
- Do not label chronic, remote, old, stable, unchanged, postoperative expected change, congenital variant, hypoplastic/aplastic/fetal variant, artifact, incidental stenosis/plaque, or collateral/reconstitution-only findings unless the same report also clearly calls the finding acute, recent, new, thrombus, occlusion, cutoff, or tissue at risk.
- Do not add secondary labels from mechanism or broad lobe wording. Add a second label only when that second territory has its own explicit vessel, infarct, ischemia, diffusion, or perfusion evidence.

Self-check:
- Labels must match the reasoning exactly.
- The final label array is not a separate guess; it must be the exact label set defended by the reasoning sentence.
- If the reasoning says "only", "excluded", "omitted", "does not qualify", or "not enough", update the final labels so they match that statement.
- If reasoning says MCA/M1/M2/insula/operculum/basal ganglia/corona radiata/centrum semiovale, output RMCA or LMCA.
- If output includes RACA/LACA, reasoning must cite ACA/A1/A2/A3 or ACA-specific anatomy. Broad "frontal", "posterior frontal", "frontoparietal", or "corona radiata/centrum semiovale" wording is not enough for ACA.
- If output includes RPCA/LPCA, reasoning must cite PCA/P1/P2/P3/P4 or PCA-specific anatomy.
- If output includes RPICA/LPICA, reasoning must cite PICA or inferior cerebellar/PICA-specific anatomy.
- If output includes RVA/LVA, reasoning must cite the right/left vertebral artery or a right/left vertebral artery segment such as V1/V2/V3/V4; do not infer RVA/LVA from generic posterior-circulation wording alone.
- If output includes RICA/LICA in CT or CTP, the reasoning must explain why the report is an unseparated ICA/carotid-territory pattern rather than a specific MCA/ACA tissue/perfusion territory.
- Do not output RCA/LCA unless those labels are present in the allowed-label list.
""".strip()


def concise_reasoning_rule(label_field: str = "labels") -> str:
    return f"""
Reasoning style:
- Do not debate multiple possible interpretations.
- Use 1 to 3 short sentences only.
- Cite only the strongest qualifying or disqualifying report phrase.
- If the final answer is ["NONE"], state the single reason no finding qualifies.
- Do not mention labels that are not in {label_field}, unless explaining why {label_field} is ["NONE"].
- The reasoning must not end with a different conclusion than {label_field}.
""".strip()


@lru_cache(maxsize=1)
def _load_optimized_prompts() -> Dict[str, Any]:
    """Load DSPy-optimized prompt guidance if it has been generated.

    The optimizer writes a small JSON file with per-modality guidance text and
    selected few-shot examples.  Prompt builders keep working when that file is
    absent, malformed, or disabled in config.py.
    """
    if not bool(getattr(cfg, "USE_OPTIMIZED_PROMPTS", True)):
        return {}

    raw_path = getattr(cfg, "OPTIMIZED_PROMPTS_FILE", None)
    if raw_path is None:
        return {}

    path = Path(raw_path)
    if not path.exists() or not path.is_file():
        return {}

    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return {}

    return data if isinstance(data, dict) else {}


def optimized_prompt_guidance(modality: str) -> str:
    """Return per-modality guidance generated by DSPy prompt optimization."""
    data = _load_optimized_prompts()
    modalities = data.get("modalities", {}) if isinstance(data, dict) else {}
    entry = modalities.get(str(modality).upper(), {}) if isinstance(modalities, dict) else {}
    if not isinstance(entry, dict) or not entry.get("enabled", True):
        return ""

    guidance = str(entry.get("guidance_text") or entry.get("instructions") or "").strip()
    if not guidance:
        return ""

    return f"""
DSPy-optimized guidance for {str(modality).upper()}:
{guidance}
""".strip()


def build_ct_sanitization_prompt(case_id: str, ct_report: str) -> str:
    return f"""
You are cleaning a report that is supposed to be ONLY a non-contrast CT head/brain report.

Your job:
1. Detect whether the supplied CT report contains CTA/CT angiogram or CTP/perfusion findings mixed into it.
2. If contamination is present, remove ONLY the CTA/CTP/perfusion/angiographic findings and return a sanitized CT-only report.
3. If no contamination is present, return the original CT report exactly as the sanitized_report.

What counts as CTA/CTP contamination:
- Sentences or sections labeled CTA, CT ANGIOGRAM, CT angiography, angiographic, arterial phase, vessel postprocessing, MIP, 3D reconstruction, circle of Willis vessel evaluation, head/neck vessel evaluation.
- Vessel-only CTA findings such as occlusion, stenosis, thrombus, filling defect, flow cutoff, absent opacification, reconstitution, collateral flow, or delayed filling when they are clearly from CTA/angiography rather than a noncontrast CT hyperdense vessel sign.
- CT perfusion findings such as CTP, perfusion, Tmax, CBF, CBV, MTT, mismatch, core volume, hypoperfusion volume, penumbra, tissue at risk, RAPID, or perfusion maps.
- Impressions/recommendations that summarize CTA/CTP findings rather than CT findings.

What must be preserved:
- Noncontrast CT-visible findings, including hemorrhage, mass effect, midline shift, edema, hypodensity, loss of gray-white differentiation, ASPECTS, hyperdense MCA/vessel sign, infarct seen on CT, chronic infarcts, encephalomalacia, lacunar infarcts, atrophy, microvascular disease, hydrocephalus, and postoperative CT findings.
- CT report structure when possible: EXAMINATION, COMPARISON, TECHNIQUE, FINDINGS, IMPRESSION.
- Do not change wording of preserved CT-only findings except for minimal cleanup needed after removing contaminated sentences.
- Do not add new medical facts.

Important edge cases:
- Keep a noncontrast CT phrase like "hyperdense left MCA sign" because that is CT-visible.
- Remove a CTA phrase like "left M1 occlusion on CTA" because that is angiographic.
- Remove CTP values like "Tmax >6 seconds", "CBF <30%", "mismatch volume", or "hypoperfusion".
- If the entire supplied text is actually CTA/CTP and no CT-only findings remain, set sanitized_report to "No noncontrast CT-only findings are provided in the supplied report."

Before returning the sanitized CT report, scan it for CTA/CTP modality terms.
The final New_CT_Report must not contain:
CTA, CT angiogram, CT angiography, CTP, CT perfusion, perfusion, Tmax, CBF, CBV, mismatch, hypoperfusion, penumbra, infarct core, tissue at risk.

If a sentence contains both a CT-visible finding and CTA/CTP wording, preserve only the CT-visible finding and remove the CTA/CTP wording.
Example:
"There is hyperdensity in the right MCA compatible with subtotal occlusion seen on the CT angiogram"
should become:
"There is hyperdensity in the right MCA compatible with thrombus."

Case ID:
{case_id}

Original CT Report:
{ct_report}

Return exactly this JSON structure:
{{
  "case_id": "{case_id}",
  "contamination_found": true,
  "sanitized_report": "CT-only report text after removing CTA/CTP contamination, or the original report if no contamination was found",
  "removed_sections": ["short description or exact removed CTA/CTP sentence"],
  "reasoning": "Brief explanation of what was removed or why no removal was needed"
}}
""".strip()


def build_ct_prompt(case_id: str, ct_report: str) -> str:
    return f"""
You are labeling ONLY the non-contrast CT report for acute stroke territory.

{base_rules()}

{optimized_prompt_guidance("CT")}

For CT-only labeling, ignore any sentence that explicitly references CTA, CT angiogram, CTP, perfusion, Tmax, CBF, CBV, mismatch, hypoperfusion, or concurrent angiographic/perfusion imaging. Use only noncontrast CT-visible findings such as hypodensity, loss of gray-white differentiation, ASPECTS, hemorrhage, mass effect, and hyperdense vessel signs.

Required CT step-by-step decision process:
- Step 1: Build an internal candidate list from every CT-visible finding that could represent acute/recent ischemic stroke. A candidate needs a side, a mappable anatomic location/territory, and CT-visible evidence such as acute/recent infarct or ischemia, new infarct, acute territorial hypodensity, loss of gray-white differentiation, edema/cytotoxic edema, ASPECTS abnormality, or a definite hyperdense MCA/M1/M2/dense-vessel sign.
- Step 2: For each candidate, check whether the same sentence, clause, or impression phrase actually supports acute/recent ischemic territory labeling. Eliminate candidates supported only by chronic/old/remote/stable/unchanged/known/evolving-follow-up/subacute-only wording, postoperative/moyamoya/bypass-related change, hemorrhage/contusion/subdural/subarachnoid/extra-axial blood without explicit infarct/ischemia, artifact, weak/questionable/subtle/possible wording with a negative acute impression, or CTA/CTP-only evidence.
- Step 3: Map only the surviving CT candidates to territories. Deep structures such as basal ganglia, lentiform, putamen, caudate, internal capsule, corona radiata, centrum semiovale, insula, and operculum map to MCA. Occipital, calcarine, posterior temporal, and thalamic findings map to PCA. Medial/parafalcine/parasagittal/cingulate/corpus-callosum/pericallosal/callosomarginal findings map to ACA. Pontine/brainstem findings described as basilar-territory map to BA. Inferior or posterior-inferior cerebellar findings map to PICA.
- Step 4: Apply conservative CT tie-breakers after elimination. If the report says no acute infarct/no acute intracranial abnormality and the only candidate is weak, subtle, questionable, or possible, return ["NONE"]. If a label is anatomically possible but not directly supported by CT wording, eliminate it. If two label sets seem possible, choose the smaller label set directly supported by explicit CT wording.
- Step 5: Return only the labels that survive elimination. If no candidate survives, return ["NONE"]. Do this candidate/elimination process internally; final reasoning should briefly name the kept label(s) or the single strongest reason nothing qualifies, without printing a long candidate ledger or self-correction.

CT candidate-elimination examples:
- "No acute intracranial abnormality" plus "questionable/subtle hypodensity" -> eliminate the weak candidate and return ["NONE"].
- "Acute left frontal corona radiata infarct" -> candidate LMCA survives because corona radiata maps to MCA, not ACA.
- "Recent infarct of the left occipital lobe" -> candidate LPCA survives.
- "Left temporal-parietal-occipital recent infarcts" -> temporal/parietal components support LMCA and occipital supports LPCA; do not add LACA unless ACA/medial/parafalcine/cingulate/corpus-callosum wording is present.
- "Subdural hematoma/subarachnoid hemorrhage with nearby subtle hypodensity" -> eliminate the ischemic-territory candidate unless the CT explicitly calls it acute infarct or acute ischemia.
- "Evolving known right basal ganglia infarct" on a follow-up CT -> eliminate as known/evolving follow-up unless the same CT clearly states a new acute infarct.

CT-specific rules:
- Use only the CT report below. Do not use CTA, CTP, MRI, symptoms, or later reports.
- CT subtle-hypodensity / hemorrhage-context rule: do not label a territory from "subtle hypodensity", "questionable hypodensity", or "possible infarct" unless the report also calls it acute/recent/new infarct, acute ischemia, loss of gray-white differentiation, or a definite acute territorial hypodensity.
- If the report's definite acute finding is hemorrhage, subdural hematoma, subarachnoid hemorrhage, contusion, or extra-axial blood, do not convert nearby subtle hypodensity into an ischemic territory label unless the report explicitly calls it acute infarct/ischemia.
- First decide whether the CT report is truly acute-positive. If the CT only describes subacute, evolving, known, chronic, remote, old, postoperative, moyamoya/bypass-related, or acute-on-chronic change, output ["NONE"] unless the same CT impression clearly says there is a new acute infarct.
- Do not convert a CT follow-up phrase like "evolving right basal ganglia infarct" into RMCA/LMCA when it is clearly a known prior infarct, follow-up exam, chronic lesion, or subacute-only finding.
- Evolving acute-to-subacute CT findings can qualify when the report presents them as the current suspected stroke event, gives a clear side/territory, and does not frame them as purely chronic, remote, postoperative, or known follow-up change.
- Label CT when the CT itself describes acute/recent ischemia, acute/recent infarct, acute territorial hypodensity, loss of gray-white differentiation, edema, ASPECTS abnormality, definite hyperdense MCA/M1/M2 thrombus, or definite dense vessel sign with a clear side and territory.
- A definite hyperdense MCA/M1/M2 sign can represent very early MCA stroke even before visible tissue damage, especially when the report uses thrombus/acute wording.
- Hyperdense-vessel tie-breaker: definite hyperdense MCA/M1/M2 -> MCA; weak/questionable hyperdensity with a negative acute impression -> NONE; hyperdense ICA/carotid terminus alone should not add RICA/LICA when CT tissue/vessel evidence is already better represented as MCA and there is no ACA tissue involvement.
- Do not add RICA/LICA from a hyperdense ICA/carotid terminus sign when the CT-visible stroke pattern is better represented by downstream MCA and there is no ACA tissue involvement.
- CT can label possible/recent ischemic change when the CT impression gives a specific side and anatomic territory, for example "possible recent ischemic change in the left occipital lobe" -> LPCA.
- Definite recent/acute infarct of the occipital lobe maps to PCA even if the CT wording is concise. Example: "recent infarct of the left occipital lobe" -> ["LPCA"].
- CT temporal-parietal-occipital or lateral parietal/temporal plus occipital wording should not create ACA. Map lateral temporal/parietal components to MCA and occipital/posterior temporal components to PCA; add ACA only for ACA/A1/A2/A3, medial, parasagittal, parafalcine, cingulate, corpus callosum, pericallosal, or callosomarginal wording.
- CT deep-structure rule: acute/recent/questionable acute infarct in corona radiata, centrum semiovale, internal capsule, basal ganglia, putamen, caudate, lentiform nucleus, insula, or operculum maps to MCA, even if the report also says frontal/frontoparietal or lacunar.
- Explicit example: "questionable small acute nonhemorrhagic left frontal corona radiata lacunar infarct" -> ["LMCA"], not ["LACA"]. "Left posterior frontal centrum semiovale infarct" -> ["LMCA"], not ["LACA"].
- CT ACA rule: do not use ACA for frontal or frontoparietal wording unless the CT report says ACA territory, medial frontal/parietal, parafalcine, parasagittal, cingulate, corpus callosum, pericallosal, callosomarginal, A1/A2/A3. Lateral frontal, lateral parietal, temporal, insula, operculum, and basal ganglia remain MCA.
- CT ICA/carotid rule: CT labels should usually be parenchymal territories. Do not add RICA/LICA from a noncontrast CT hyperdense carotid terminus/ICA thrombus when the CT already identifies downstream MCA/ACA territory infarct.
- Specific-over-ICA CT example: CT says "entire left MCA and ACA infarct" plus "hyperdense thrombus at left carotid terminus" -> ["LMCA", "LACA"], not ["LICA", "LMCA", "LACA"].
- If CT mentions ICA/carotid terminus but only one downstream territory is definite, do not invent the missing ACA/MCA territory; use only the definite evidence.
- Do NOT label an isolated weak vessel-sign phrase such as "question of mild hyperdensity" when the same CT impression says no acute cortical infarct/no acute intracranial abnormality. In that situation, output ["NONE"] for CT.
- Do NOT label a known/evolving/subacute infarct that is only being followed from a prior CT/MRI if the report does not describe a new acute CT abnormality. Phrases like "evolving recent infarct", "evolving basal ganglia infarct", "subacute lacunar infarct", or "known infarct" should be treated as follow-up evidence, not a new CT-positive territory.
- Do NOT label suspected acute-on-chronic border-zone/moyamoya/bypass changes unless the CT impression clearly identifies a new acute territorial infarct separate from the chronic disease.
- Explicit example: moyamoya with "suspected acute on chronic right MCA territory ischemic changes", chronic bypass/synangiosis, or progressed border-zone hypoattenuation -> ["NONE"] for acute-only CT scoring unless a separate new acute infarct is clearly stated.
- Do NOT label chronic lacunar infarcts, chronic microvascular disease, encephalomalacia, old infarcts, or postoperative/moyamoya/bypass-related chronic changes.
- In moyamoya or bypass cases, do not label CT from chronic border-zone changes, graft/bypass discussion, or acute-on-chronic wording unless the CT clearly identifies a new acute infarct separate from chronic disease.
- Output ["NONE"] when CT says no acute infarct/hemorrhage and only describes chronic/incidental findings.

Case ID:
{case_id}

CT Report:
{ct_report}

{final_consistency_check("labels")}

{concise_reasoning_rule("labels")}

Return exactly this JSON structure:
{{
  "case_id": "{case_id}",
  "modality": "CT",
  "labels": ["<LABELS>"],
  "reasoning": "CT-only reason supporting the final labels"
}}
""".strip()


def build_cta_prompt(case_id: str, cta_report: str) -> str:
    return f"""
You are labeling ONLY CTA vessel abnormalities.

Allowed labels:
{labels_text()}

{optimized_prompt_guidance("CTA")}

Output rules:
- Return JSON only. No markdown, comments, or extra text.
- labels must contain at least one label.
- Use ["NONE"] when the CTA report gives no qualifying vessel abnormality.
- If any territory label is present, do not include "NONE".
- If multiple labels are needed, use this order: {label_order_text()}.
- Use only the CTA report below. Do not use CT, CTP, MRI, symptoms, or downstream infarct/perfusion to create CTA labels.
- Do not override the rules below because a vessel "may account for ischemia" or because the patient has stroke symptoms.

Required CTA step-by-step decision process:
- Step 1: Build an internal candidate list from every named vessel/territory in the CTA report. A candidate needs a side, a mappable vessel/territory, and CTA wording that could qualify.
- Step 2: For each candidate, check whether the same sentence, clause, or impression phrase gives qualifying CTA evidence: definite occlusion, thrombus, embolus/clot, filling defect, flow cutoff, absent opacification, nonopacification/nonfilling/nonvisualization, near-occlusion, or definite diminished distal flow from a named branch abnormality.
- Step 3: Eliminate each candidate that is stenosis-only, narrowing-only, atherosclerosis/plaque-only, chronic/stable/known/old, postoperative collateral/reconstitution-only, hypoplastic/aplastic/fetal/variant anatomy, limited-evaluation/cannot-exclude wording, or uncertain "stenosis versus occlusion" wording without a separate definite occlusion/thrombus/filling-defect/cutoff phrase.
- Step 4: After eliminations, apply the ICA rule. Keep RICA/LICA when ICA/carotid occlusion is definite and downstream MCA/ACA evidence is not both independently definite. Drop RICA/LICA only when same-side MCA and ACA are both independently definite.
- Step 5: Return only the labels that survive elimination. If no candidate survives, return ["NONE"].
- Do this candidate/elimination process internally. In the final reasoning, briefly name the kept label(s) and only mention eliminated labels when they are the main reason a tempting label was not included. Do not output a long debate, self-correction, or "wait" reasoning.

CTA candidate-elimination example:
- "Complete occlusion of the left internal carotid" -> candidate LICA survives.
- "Left M1 severe stenosis versus complete occlusion" without a separate definite thrombus/filling defect/cutoff phrase -> candidate LMCA is eliminated for uncertainty.
- "Hypoplastic left A1" -> candidate LACA is eliminated as variant anatomy.
- "Right A2 atherosclerotic disease" and "20% right P2 stenosis" -> RACA/RPCA are eliminated as stenosis/atherosclerosis-only.
- Final CTA label in that pattern is ["LICA"].

Qualifying CTA findings:
Label the named vessel/territory when CTA explicitly states any of the following:
- occlusion, occluded, complete occlusion, segmental occlusion, short-segment occlusion
- "possible short-segment occlusion" or "may be a short segment thrombus" in a named artery segment, when not contradicted by a negative impression. Named short-segment PCA thrombus/occlusion phrases qualify even with "possible" or "may be" when the vessel segment is specific and the finding is not contradicted elsewhere.
- "stenosis to occlusion" / "stenoses to occlusions" / "severe narrowing or segmental occlusion"
- thrombus, thrombosis, embolus, clot, filling defect, tandem filling defects when the report presents them as definite rather than merely possible/concern/question
- flow cutoff/cutoff, absent opacification, nonopacification, nonfilling, nonvisualization/not visualized, no flow
- near-occlusion/near-occlusive/subocclusive thrombus
- definite nonocclusive thrombus still qualifies when the report clearly says thrombus/clot/filling defect in a named artery; possible/concern-for/questionable nonocclusive thrombus does not qualify
- definite diminished distal flow or delayed distal filling caused by a named branch abnormality

CTA findings that are NOT enough by themselves:
- CTA secondary-label rule: add a second CTA label only when that vessel/territory has its own definite acute occlusion, thrombus, cutoff, absent opacification, near-occlusion, or filling defect.
- Do not add a secondary CTA label for stenosis, mild/moderate/severe narrowing, luminal irregularity, plaque, fetal origin, hypoplastic/aplastic variants, collateral flow, or reconstitution alone.
- If the report says "may be thrombus", "possible thrombus", or "questionable occlusion", treat it as uncertain unless the impression confirms it as a real acute thrombus/occlusion. The existing named short-segment vessel exception still applies only when the segment is specific and not contradicted.
- stenosis, narrowing, irregularity, plaque, atherosclerosis, calcification, mild/moderate/severe/high-grade stenosis
- severe stenosis without any occlusion/thrombus/cutoff/filling-defect/nonopacification wording
- "severe stenosis", "critical stenosis", "high-grade stenosis", or "marked narrowing" alone should be NONE for CTA, including basilar stenosis, unless the same sentence/report also says thrombus, clot, filling defect, occlusion, cutoff, absent/nonopacification, or near-occlusion.
- "possible nonocclusive thrombus", "concern for nonocclusive thrombus", "correlate if concern for thrombus", or "may or may not account for ischemia". This exclusion is for nonocclusive/unclear thrombus only; it does not exclude a named "possible short-segment occlusion" or "may be short-segment thrombus" in P1/P2/P3/P4/M1/M2/A1/A2.
- General uncertainty alone is not enough, but a phrase that still names a short-segment occlusion/thrombus in a specific vessel segment can qualify.
- "limited evaluation for occlusion", "cannot exclude occlusion", "evaluation for partial occlusion is limited", or "stenosis versus occlusion" when there is no separate definite filling defect/thrombus/occlusion phrase
- chronic collateral flow, reconstitution, hypoplastic/aplastic/fetal variants, and incidental plaque
- an impression saying "no acute intracranial arterial vascular abnormality" should force ["NONE"] unless the same impression separately states a qualifying occlusion/thrombus/cutoff/filling defect.

Chronic/stable nuance:
- Do not label an occlusion described as known/similar/stable chronic occlusion, especially in moyamoya or postsurgical bypass/synangiosis cases.
- In moyamoya/vasculopathy cases, new or progressive stenosis is still stenosis. Do not label it unless the report explicitly says occlusion, thrombus, filling defect, cutoff, or "stenosis to occlusion".
- Do label a named branch when the report says definite segmental occlusion or "severe narrowing or segmental occlusion" and it is not specifically described as stable chronic collateral disease.

Vessel mapping:
- Finding-side rule: assign the side from the same finding sentence, clause, or impression phrase. Do not transfer the side from one finding to a different finding elsewhere in the report. However, bilateral or multifocal strokes are allowed: if the report gives separate qualifying acute evidence on both the left and right sides, include both left-sided and right-sided labels. 
- Right MCA/M1/M2/M3/M4 -> RMCA; left MCA/M1/M2/M3/M4 -> LMCA.
- Right ACA/A1/A2/A3/pericallosal/callosomarginal -> RACA; left ACA/A1/A2/A3/pericallosal/callosomarginal -> LACA.
- Right PCA/P1/P2/P3/P4 -> RPCA; left PCA/P1/P2/P3/P4 -> LPCA.
- Right PICA -> RPICA; left PICA -> LPICA.
- Right vertebral artery/V1/V2/V3/V4/intradural vertebral artery -> RVA.
- Left vertebral artery/V1/V2/V3/V4/intradural vertebral artery -> LVA.
- Right ICA/internal carotid/petrous/cavernous/paraclinoid/supraclinoid/intracranial ICA/carotid terminus -> RICA only when ICA is not superseded by more specific same-side MCA + ACA labels.
- Left ICA/internal carotid/petrous/cavernous/paraclinoid/supraclinoid/intracranial ICA/carotid terminus -> LICA only when ICA is not superseded by more specific same-side MCA + ACA labels.
- Do not infer MCA/ACA/PCA from ICA alone. Add downstream branch labels only when those branches have their own CTA occlusion/thrombus/cutoff/filling-defect evidence.
- Specific-over-ICA CTA rule: if ICA/carotid terminus thrombus/occlusion is the upstream/general source and the report separately identifies both same-side MCA/M1/M2 and ACA/A1/A2/A3 involvement, output only the downstream MCA + ACA labels and omit RICA/LICA.
- Explicit example: left carotid terminus/ICA thrombus plus occluded left MCA and occluded left A1/proximal ACA -> ["LMCA", "LACA"], not ["LICA", "LMCA", "LACA"].
- If there is an ICA/paraclinoid/supraclinoid/cavernous ICA occlusion plus MCA occlusion but no definite ACA occlusion, keep both ICA and MCA when both are definite.
- Do not apply the MCA+ACA-over-ICA rule to ICA+MCA-only cases. Definite ICA/carotid occlusion plus definite MCA occlusion with only variant/uncertain A1 anatomy -> ["RICA", "RMCA"] or ["LICA", "LMCA"] depending on side.
- Do not treat hypoplastic/aplastic/fetal/variant A1/A2 anatomy as acute ACA involvement. A hypoplastic or possibly occluded A1 variant does not trigger the MCA+ACA-over-ICA rule.
- If the report says left MCA, left M1, left M2, left M3, or left M4, the label must be LMCA.
- If the report says right MCA, right M1, right M2, right M3, or right M4, the label must be RMCA.
- For cervical carotid/common carotid disease, output RICA/LICA only if the report specifically says ICA/internal carotid; otherwise do not force it into an unavailable RCA/LCA label.

Side and territory self-check:
- Right A1/A2/A3/ACA must be RACA, never LACA.
- Left A1/A2/A3/ACA must be LACA, never RACA.
- Right P1/P2/P3/P4/PCA must be RPCA, never LPCA.
- Left P1/P2/P3/P4/PCA must be LPCA, never RPCA.
- Right vertebral artery/V1/V2/V3/V4 must be RVA, never LVA.
- Left vertebral artery/V1/V2/V3/V4 must be LVA, never RVA.
- Vertebral artery labels require definite occlusion/thrombus/cutoff/filling defect/nonopacification/near-occlusion. Vertebral stenosis or narrowing alone is not RVA/LVA.
- PCA labels require occlusion/thrombus/cutoff/filling defect/segmental occlusion. P2/P3 stenosis or narrowing alone is not RPCA/LPCA.
- "possible short-segment occlusion" or "may be a short segment thrombus" of P1/P2/P3/P4 qualifies as RPCA/LPCA; "stenosis/narrowing" alone does not. Example: "may be a short segmental thrombus in the left P2 PCA" -> ["LPCA"].
- If reasoning says a finding is "not enough", "does not qualify", "stenosis alone", or "limited evaluation", the final label for that finding must be omitted.

Examples:
- "moderate stenosis of left M2" -> ["NONE"]
- "moderate luminal narrowing of a distal right M2/M3 branch, may or may not account for ischemia" -> ["NONE"]
- "moderate-severe narrowing of left P2" -> ["NONE"]
- "severe focal stenosis of left P2 PCA" + "no acute arterial vascular abnormality" -> ["NONE"]
- "evaluation for right M1/M2 occlusion is limited" -> ["NONE"]
- "concern for nonocclusive acute thrombus" without definite thrombus/filling defect -> ["NONE"]
- "right M1 severe narrowing and possible short segment occlusion" -> ["RMCA"]
- "left P2 may be a short segment thrombus" -> ["LPCA"]
- "right P2 possible short-segment occlusion" -> ["RPCA"]
- "occlusion of the right vertebral artery" -> ["RVA"]
- "nonopacification of the left V4 vertebral artery" -> ["LVA"]
- "left carotid terminus thrombus with left MCA and left A1 ACA occlusions" -> ["LMCA", "LACA"]
- "stable chronic occlusion of right M1 in moyamoya" -> ["NONE"]
- "stable severe narrowing or segmental occlusion of left M1" -> ["LMCA"]
- "occlusion of right A2 ACA" -> ["RACA"]
- "complete occlusion of the right paraclinoid/intracranial ICA and right MCA, with right A1 hypoplastic/variant or cross-filled and no definite right ACA occlusion" -> ["RICA", "RMCA"]
- Do not omit RICA/LICA merely because the ACA is not affected or is cross-filled; that situation means there is no definite ACA label, so ICA + MCA should be kept when both ICA and MCA are definitely occluded.

CTA multi-vessel scan:
- Before returning labels, scan the CTA report for every qualifying vessel abnormality, not just the first or dominant abnormality.
- Build candidate labels separately for:
  1. ICA/carotid terminus/internal carotid
  2. MCA/M1/M2/M3/M4
  3. ACA/A1/A2/A3/pericallosal/callosomarginal
  4. PCA/P1/P2/P3/P4
  5. PICA
  6. vertebral artery/V1/V2/V3/V4
- Include every candidate with definite qualifying CTA evidence unless excluded by chronic/stable, stenosis-only, variant, limited-evaluation, or nonocclusive-uncertain wording.
- Do not drop a definite ACA, PCA, PICA, or vertebral artery occlusion because a larger MCA/ICA/basilar abnormality is also present.
- Do not infer downstream MCA/ACA/PCA/PICA from ICA/carotid or vertebral artery disease alone. Downstream labels require their own CTA evidence.

Final CTA label reconciliation:
- Before returning the final CTA labels, compare the reasoning against the labels array.
- If the CTA report directly describes a definite occlusion, thrombus, filling defect, flow cutoff, absent opacification, or near-occlusion in the ICA/internal carotid/carotid terminus, the matching RICA/LICA label must be included unless the same-side MCA and ACA are both independently definite CTA-positive.
- Do not drop RICA/LICA only because a downstream MCA or ACA abnormality is also present.
- Drop RICA/LICA only when both same-side downstream territories are independently definite, such as definite MCA/M1/M2 involvement and definite ACA/A1/A2/A3 involvement.
- Do not treat uncertain, variant, hypoplastic, aplastic, fetal-origin, collateral-supplied, limited-evaluation, or ambiguous branch wording as definite downstream territory involvement.
- If only one same-side downstream territory is definite, keep both the ICA/carotid label and the definite downstream territory label.
- A hypoplastic, variant, cross-filled, contralateral-supplied, collateral-supplied, or uncertain ACA/A1/A2 does not count as definite ACA involvement and must not be used to drop RICA/LICA.
- If the CTA report directly describes a definite right or left vertebral artery/V1/V2/V3/V4 occlusion, thrombus, filling defect, flow cutoff, absent opacification, nonopacification, or near-occlusion, the matching RVA/LVA label must be included unless it is explicitly chronic/stable or stenosis-only.
- The final labels must match the reasoning. If the reasoning mentions a definite qualifying ICA/carotid abnormality but the labels omit RICA/LICA, revise the labels before output. If the reasoning mentions a definite qualifying vertebral abnormality but the labels omit RVA/LVA, revise the labels before output.

Case ID:
{case_id}

CTA Report:
{cta_report}

{final_consistency_check("labels")}

{concise_reasoning_rule("labels")}

Return exactly this JSON structure:
{{
  "case_id": "{case_id}",
  "modality": "CTA",
  "labels": ["<LABELS>"],
  "reasoning": "CTA-only reason supporting the final labels"
}}
""".strip()


def build_ctp_prompt(case_id: str, ctp_report: str) -> str:
    return f"""
You are labeling ONLY the CTP/perfusion report for acute stroke territory.

{base_rules()}

{optimized_prompt_guidance("CTP")}

Required CTP step-by-step decision process:
- Step 1: Build an internal candidate list from every side-specific perfusion/core/tissue-at-risk finding and every clearly quoted acute vessel/territory finding inside the CTP report.
- Step 2: For each candidate, require CTP evidence tied to that same side/territory: true hypoperfusion, Tmax delay, mismatch, penumbra, tissue at risk/brain at risk, reduced CBF, infarct core, or a clearly quoted acute occlusion/thrombus/cutoff in the CTP text.
- Step 3: Eliminate each candidate that is tiny/nonspecific (<10 mL Tmax/mismatch with 0 mL core and relative/nonspecific wording), artifact-only, chronic/stigmata-of-moyamoya/collateral/bypass-related, global/bilateral/nonspecific without separate side-specific territory evidence, or unsupported by a named/mapped territory.
- Step 4: Apply mapping after eliminations. Named MCA/ACA/PCA/PICA/BA perfusion territories beat broad mechanism. Use RICA/LICA only when the CTP report frames the perfusion abnormality as an unseparated ICA/carotid-territory pattern and does not separate MCA/ACA territories.
- Step 5: Return only surviving candidates. If no candidate survives, return ["NONE"].
- Do this candidate/elimination process internally. In the final reasoning, briefly name the kept label(s) and only mention eliminated labels when needed. Do not output a long debate, self-correction, or "wait" reasoning.

CTP-specific rules:
- Apply literal report-text rules before broader anatomy inference.
- Use the named perfusion territory/anatomy as the anchor. Do not split a large volume into many territories unless the CTP report explicitly names each territory or gives side-specific anatomy for each one.
- CTP side-lock rule: right-sided perfusion wording maps only to right labels and left-sided perfusion wording maps only to left labels. Do not mirror a label to the opposite side unless the report says bilateral or separately describes the opposite side.
- When CTP wording is broad, use these tie-breakers: lateral frontal/parietal/temporal/frontoparietal -> MCA; medial/parafalcine/parasagittal/corpus callosum/cingulate -> ACA; occipital/posterior temporal/thalamus -> PCA; inferior/posterior-inferior cerebellum or named PICA -> PICA; pons/brainstem/basilar-territory -> BA.
- CTP ACA guard: temporal-parietal-occipital or lateral parietal/temporal plus occipital wording should not create ACA. Map lateral temporal/parietal components to MCA and occipital/posterior temporal components to PCA; add ACA only for ACA/A1/A2/A3, medial, parasagittal, parafalcine, cingulate, corpus callosum, pericallosal, or callosomarginal wording.
- CTP PCA threshold rule: a localized Tmax >6 seconds hypoperfusion/mismatch volume of 10 mL or greater in an occipital or parietal-occipital region can qualify as PCA even when infarct core is 0 mL, unless the report calls it artifact, relative/nonspecific, clinically insignificant, or gives no side/territory localization.
- Broad phrases such as "posterior circulation", "cerebral hemisphere", "multifocal", or a large RAPID volume are not enough by themselves to add every possible PCA/PICA/BA/MCA/ACA label.
- CTP bilateral/global perfusion rule: do not label both MCA territories from broad bilateral/global hypoperfusion unless the report explicitly names both left and right MCA territories or gives separate left-sided and right-sided core/penumbra/hypoperfusion evidence.
- If the report gives one dominant named territory and vague contralateral/global hypoperfusion, label only the dominant named territory.
- If Tmax/core/mismatch values are global or nonspecific without a vascular territory, do not infer MCA/PCA/ACA labels.
- If the CTP text uses malformed or shorthand wording such as "left-sided cord infarct" with left-sided core/hypoperfusion and gives no better named anterior-circulation territory, treat it as a left posterior-fossa/PICA-type localization rather than inferring MCA from volume alone.
- If that malformed left-sided posterior-fossa/cord-like wording is the only territorial clue, label LPICA instead of inferring LMCA from a large nonspecific left-sided volume.
- CTP PICA guard: do not add RPICA/LPICA from nonspecific "cerebellar hemisphere" wording alone unless the report says PICA, posterior-inferior cerebellum, inferior cerebellum, or clearly posterior-inferior cerebellar territory. Side-lock PICA findings: left cerebellar/PICA evidence cannot support RPICA, and right cerebellar/PICA evidence cannot support LPICA.
- Use only the CTP report below. Do not use CT, CTA, or MRI except when the CTP report itself quotes them.
- Label the territory of true hypoperfusion, Tmax delay, mismatch, penumbra, tissue at risk, brain at risk, reduced CBF, or infarct core when a side and territory are clear.
- 0 mL infarct core does not mean NONE if there is a meaningful named hypoperfusion/penumbra/tissue-at-risk territory.
- Do not output NONE when the CTP report gives a substantial nonzero hypoperfusion/Tmax delay/mismatch volume and localizes it to a named territory such as right MCA or left MCA, even if the infarct core is 0 mL or the impression says no definite core infarct.
- If the CTP impression links the finding to recent ischemia in a named territory on concurrent CT and the CTP has a large hypoperfusion volume, label that named CTP territory.
- If CTP explicitly says "right MCA territory" or "left MCA territory", output RMCA/LMCA. However, if the same CTP report also quotes or mentions same-side ACA/A1/A2/A3 occlusion, thrombus, cutoff, or absent opacification in the acute stroke context, include the ACA label too.
- If CTP explicitly identifies both MCA and ACA perfusion/core abnormalities on the same side, output MCA + ACA and omit ICA even if an ICA/carotid terminus source is mentioned.
- If the CTP report itself frames the entire perfusion abnormality as ICA/internal carotid/carotid-territory and does not separate MCA/ACA territories, use RICA/LICA for CTP.
- If the CTP report frames a large ipsilateral anterior-circulation perfusion deficit as an ICA/carotid terminus pattern and does not provide separate, clearly localized MCA and ACA perfusion territories, use RICA/LICA instead of splitting into MCA+ACA.
- Explicit example: CTP frames the deficit only as left ICA/carotid-territory hypoperfusion/penumbra/core -> ["LICA"]. But if CTP localizes left MCA core/penumbra and left ACA hypoperfusion separately -> ["LMCA", "LACA"], not ["LICA"].
- For frontoparietal perfusion deficits, map to MCA unless the CTP report explicitly says ACA, medial, parafalcine, parasagittal, cingulate, corpus callosum, or A1/A2/A3.
- Occipital hypoperfusion maps to PCA. If the report says Tmax >4 seconds hypoperfusion at the left occipital lobe and recommends MRI correlation, label LPCA even if Tmax >6 seconds is 0.
- Do not infer ICA from CTP just because a CTA vessel is upstream. CTP labels should be perfusion territories, not mechanism labels.
- If the CTP report itself quotes or mentions acute ICA terminus/carotid terminus disease together with same-side MCA/M1/M2 and ACA/A1/A2/A3 occlusion, thrombus, cutoff, or absent opacification, include the specific downstream branch-territory labels, such as ["LMCA", "LACA"] or ["RMCA", "RACA"], even if the perfusion/core wording mainly says MCA territory.
- Do NOT label tiny/nonspecific isolated CTP findings: if Tmax >6 s or mismatch is under 10 mL and infarct core is 0 mL, output ["NONE"] unless the report explicitly calls it a real infarct core, penumbra, tissue at risk, or brain at risk. "Relative increased Tmax" or a tiny "small area" is not enough even if it names an MCA territory.
- Explicit example: 6 mL right parietal-temporal Tmax delay / "relative increased Tmax" / 0 mL core -> ["NONE"], not ["RMCA"].
- Low-threshold Tmax >4 s alone is usually not enough; the exception is a named occipital/PCA-territory finding with MRI correlation language.
- In moyamoya, bypass/synangiosis, chronic multivessel disease, or chronic collateral cases, do not label perfusion deficits that are described as stigmata/chronic vascular disease unless the CTP impression clearly identifies acute infarct core or acute tissue at risk separate from chronic disease.
- If the impression says "stigmata of moyamoya disease" or ties the perfusion pattern to chronic collateral/bypass disease, output ["NONE"] even if RAPID volumes are nonzero.
- Output ["NONE"] if the report says no perfusion defect, no regions of hypoperfusion, no brain at risk, artifact only, chronic-only, or gives no side/territory clue.
- If the CTP report itself mentions ICA terminus, carotid terminus, or internal carotid occlusion AND also mentions both same-side MCA/M1/M2 and ACA/A1/A2/A3 involvement, prefer the specific downstream branch labels MCA + ACA when those branches are explicitly named.
- In that situation, do not collapse the answer to RICA/LICA if same-side MCA and ACA branch involvement are both specifically identified.

CTP decision logic:
- Malformed posterior-fossa/cord-like perfusion wording with no better named territory -> PICA territory.
- Large CTP deficit framed by the CTP report as ICA/carotid-territory disease -> RICA/LICA only when no specific MCA/ACA perfusion territory is separately named.
- If CTP names MCA territory plus mentions an upstream ICA/carotid cause, output MCA rather than adding ICA unless the perfusion abnormality itself is described as ICA/carotid-territory.
- Tiny <10 mL Tmax/mismatch with 0 mL core and relative/nonspecific wording -> ["NONE"].
- If uncertain between one territory and many, choose the explicitly named smaller label set.

Case ID:
{case_id}

CTP Report:
{ctp_report}

{final_consistency_check("labels")}

{concise_reasoning_rule("labels")}

Return exactly this JSON structure:
{{
  "case_id": "{case_id}",
  "modality": "CTP",
  "labels": ["<LABELS>"],
  "reasoning": "CTP-only reason supporting the final labels"
}}
""".strip()


def build_combined_prompt(
    case_id: str,
    ct_report: str,
    cta_report: str,
    ctp_report: str,
    mri_report: str,
    ct_labels: List[str],
    cta_labels: List[str],
    ctp_labels: List[str],
) -> str:
    return f"""
You are assigning Combined_GT across CT, CTA, CTP, and MRI.
This is a reconciliation step: do not re-label CT, CTA, or CTP from scratch.

Allowed labels:
{labels_text()}

Preliminary labels:
CT_GT = {ct_labels}
CTA_GT = {cta_labels}
CTP_GT = {ctp_labels}

{optimized_prompt_guidance("COMBINED")}

Output rules:
- Return JSON only.
- Combined_GT must contain at least one label.
- Use ["NONE"] only if no candidate survives.
- If any territory label is present, do not include "NONE".
- Use this label order: {label_order_text()}.
- Reasoning must be 1 to 3 concise sentences and must support only Combined_GT.

Required internal workflow:
1. Build candidates.
   - Start with the union of non-NONE CT_GT, CTA_GT, and CTP_GT.
   - Add MRI candidates only when MRI directly says acute infarct, recent infarct, acute ischemia, restricted diffusion/diffusion restriction, early subacute infarct, or acute territorial infarction in a named/mapped territory.
   - Do not create new CT/CTA/CTP-derived candidates from raw report text if the label is absent from CT_GT, CTA_GT, and CTP_GT.

2. Evaluate each candidate separately.
   Keep a candidate only when it has direct side-specific evidence from CT_GT, CTA_GT, CTP_GT, or direct MRI acute/recent evidence.
   Remove a candidate only when the exact supporting finding is clearly chronic/old/remote/stable/unchanged/known follow-up, postoperative, artifact, explicitly not real, weak/questionable CT with a negative acute impression, stenosis-only/uncertain CTA, tiny/nonspecific CTP, or isolated upstream ICA mechanism better represented by a surviving downstream candidate.

3. Final labels equal survivors.
   - Do not add labels from mechanism, symptoms, broad vascular supply, stenosis, vague lobe wording, or anatomic possibility.
   - If two label sets remain plausible, choose the smaller set unless every extra label has its own direct evidence.

Source-specific keep/remove rules:
- CT candidates: keep definite acute/recent infarct, acute ischemia, acute territorial hypodensity, loss of gray-white differentiation, edema/ASPECTS abnormality, or definite dense MCA/vessel sign. Remove chronic/old/evolving-known follow-up, postoperative/moyamoya-only, hemorrhage/subdural/SAH/contusion-only, or subtle/questionable/possible wording with a negative acute impression.
- CTA candidates: keep definite occlusion, thrombus/clot, filling defect, cutoff, absent/nonopacification, nonfilling/nonvisualization, near-occlusion, or definite branch abnormality. Remove stenosis/narrowing/plaque/atherosclerosis-only, variant/hypoplastic/fetal anatomy, chronic/stable/known occlusion, limited-evaluation/cannot-exclude wording, or stenosis-versus-occlusion without a separate definite qualifying phrase. Do not remove a definite CTA branch label just because CT/CTP/MRI show no tissue injury.
- CTP candidates: keep named/localized hypoperfusion, Tmax delay, core, mismatch, penumbra, tissue-at-risk/brain-at-risk, or reduced CBF. Remove artifact-only, chronic/moyamoya/collateral-only, no real perfusion defect, global/nonspecific without mapped territory, or tiny/nonspecific <10 mL Tmax/mismatch with 0 mL core and relative/nonspecific wording. Keep named MCA+ACA or watershed CTP labels unless the raw CTP clearly gives a removal reason.
- MRI candidates: add/keep only direct acute/recent/restricted-diffusion territorial findings. Do not add tiny, punctate, scattered, trace, minimal, questionable, possible, or nonspecific MRI-only secondary foci when CT/CTA/CTP already define a dominant territory, unless MRI clearly calls the extra finding a named-territory acute/recent infarct or restricted-diffusion lesion.

Mapping rules:
- MCA: MCA/M1-M4, insula, operculum, basal ganglia, lentiform, putamen, caudate, internal capsule, corona radiata, centrum semiovale, periatrial white matter, lateral frontal/parietal/temporal cortex.
- ACA: ACA/A1-A3, pericallosal, callosomarginal, medial frontal/parietal, parafalcine/parasagittal, cingulate, corpus callosum. Generic frontal/frontoparietal/parietal/corona-radiata/centrum-semiovale wording is not ACA.
- PCA: PCA/P1-P4, occipital, calcarine, posterior temporal, thalamus, parietal-occipital.
- PICA: PICA, inferior cerebellum, posterior-inferior cerebellum, or clearly posterior-inferior cerebellar territory. Generic, superior, or medial cerebellar hemisphere wording alone is not PICA.
- BA: acute pontine, central pontine, paramedian pontine, acute brainstem, basilar artery, or basilar-territory infarct/ischemia when BA is allowed.
- ICA: RICA/LICA is an upstream label. Do not split a broad ICA/carotid-distribution phrase into MCA+ACA unless MCA/ACA are already preliminary candidates or directly supported by MRI. Keep ICA when it is the only surviving candidate or downstream evidence is uncertain/variant/stenosis-only.

Special reconciliation rules:
- Explicit MRI acute named-territory infarct survives even if CT/CTA/CTP are negative or point to a different dominant side, unless it is tiny/punctate/scattered/nonspecific.
- ACA-MCA watershed, ACA-MCA territory, medial/parasagittal/parafalcine frontal, or ACA-MCA border-zone acute core/hypoperfusion/restricted diffusion supports same-side MCA + ACA when directly stated.
- Centrum semiovale, corona radiata, internal capsule, basal ganglia, insula, and operculum map to MCA, not ACA.
- Occipital or parietal-occipital acute infarct/perfusion maps to PCA.
- Acute pontine/brainstem infarct maps to BA when BA is allowed, unless chronic/old/remote/stable/artifact.

Side-lock:
- Right evidence supports only right labels; left evidence supports only left labels.
- Bilateral labels require separate evidence for both sides or explicit bilateral same-territory wording.
- Left cerebellar/PICA evidence cannot support RPICA, and right cerebellar/PICA evidence cannot support LPICA.

Before output:
- Every Combined_GT label must trace to CT_GT, CTA_GT, CTP_GT, or one direct MRI acute/recent phrase.
- Every secondary label must have its own direct evidence.
- Do not output the internal candidate ledger.

Case ID:
{case_id}

CT Report:
{ct_report}

CTA Report:
{cta_report}

CTP Report:
{ctp_report}

MRI Report:
{mri_report}

{final_consistency_check("Combined_GT")}

Return exactly this JSON structure:
{{
  "case_id": "{case_id}",
  "Combined_GT": ["<LABELS>"],
  "reasoning": "1 to 3 concise sentences supporting only the final Combined_GT labels"
}}
""".strip()
