"""Verified descriptor -> LOINC rules for high-volume clinical labs the fuzzy
matcher missed or mis-mapped (synonym gaps, %/absolute and ratio confusions).
Every code below was confirmed ACTIVE + clinically correct against LOINC v2.82.

Guards exclude non-blood specimens (CSF/body fluid), timed/24-hr/clearance urine
measures, ratios, and free fractions, so a serum/blood rule never fires on a
different-specimen or derived variant. Used by apply_clinical_seed.
"""
from __future__ import annotations
import re

# Differential leukocytes: (keyword, percent_code, absolute_code)
_DIFF = [
    ("NEUTROPHIL", "770-8", "751-8"),
    ("LYMPHOCYTE", "736-9", "731-0"),
    ("MONOCYTE", "5905-5", "742-7"),
    ("MONOS", "5905-5", "742-7"),
    ("EOSINOPHIL", "713-8", "711-2"),
    ("BASOPHIL", "706-2", "704-7"),
    ("IMMATURE GRANULOCYTE", "71695-1", "53115-2"),
    ("BAND", "35332-6", None),
]
_NONBLOOD = ("BODY FLUID", "CSF", "SYNOVIAL", "PLEURAL", "PERITONEAL", "PERICARD",
             "ASCIT", "SPINAL", "VITREOUS", "DIALYSIS", "MACROPHAGE")
_TIMED = ("24 HR", "24HR", "24 HOUR", "CLEARANCE", "SUPERSAT", "EXCRETION", "TELOPEPTIDE")


def _w(u: str, *words: str) -> bool:
    return any(re.search(rf"\b{re.escape(w)}\b", u) for w in words)


def classify(desc: str) -> str | None:
    u = str(desc).upper()
    nonblood = any(s in u for s in _NONBLOOD) or _w(u, "BF", "PF")
    timed = any(s in u for s in _TIMED) or _w(u, "RATE")
    pct = ("%" in u) or _w(u, "PCT", "PERCENT") or "/100" in u
    absol = (not pct) and ("ABSOLUTE" in u or "#" in u or _w(u, "ABS") or "COUNT" in u or "NUMBER" in u)

    # --- A1c %, HDL ratio, eGFR (handled before guards; eGFR contains 'RATE') ---
    if _w(u, "A1C") or "HGBA1C" in u or "HEMOGLOBIN A1C" in u:
        return "4548-4"
    if "CHOLESTEROL/HIGH DENSITY" in u or "CHOL/HDL" in u or "CHOLESTEROL/HDL" in u:
        return "9830-1"
    if _w(u, "EGFR") or "GLOMERULAR FILTRATION" in u:
        return "98979-8"

    # --- differential leukocytes (blood only) ---
    for kw, pc, ac in _DIFF:
        if kw in u:
            if nonblood:
                return None
            if pct:
                return pc
            if absol and ac:
                return ac
            return None
    if "RETICULOCYTE" in u and not nonblood:
        if pct or "/100" in u:
            return "17849-1"
        if absol:
            return "14196-0"
        return None

    # --- urine-specific analytes (require urine context) ---
    urine = "URINE" in u or _w(u, "UA")
    if urine:
        if "ALBUMIN" in u and "CREATININE" in u:
            return "9318-7"
        if "MICROALBUMIN" in u or "ALBUMIN" in u:
            return "1754-1"
        if "CREATININE" in u and "/" not in u and "PROTEIN" not in u and not timed:
            return "2161-8"
        if "SPECIFIC GRAVITY" in u:
            return "2965-2"
        if _w(u, "PH"):
            return "5803-2"
        return None  # other urine tests (protein, sediment) -> review

    # --- everything below is serum/blood; skip other specimens / timed / free ---
    if nonblood or timed:
        return None
    if "SPECIFIC GRAVITY" in u:
        return "2965-2"

    if _w(u, "B12") or "VITAMIN B12" in u:
        return "2132-9"
    if ("25-HYDROXY" in u or "25 HYDROXY" in u or "25-HYDROXYVITAMIN" in u) and "DIHYDROXY" not in u:
        if _w(u, "D2"):
            return "49054-0"   # 25-OH-D2 (keep separate; D2 is a distinct analyte)
        if _w(u, "D3"):
            return "1989-3"    # 25-OH-D3
        return "62292-8"       # 25-OH total (D2+D3)
    if "LIPASE" in u:
        return "3040-3"
    if (_w(u, "PSA") or "PROSTATE SPECIFIC" in u) and "FREE" not in u:
        return "2857-1"
    if "FREE T4" in u or ("T4" in u and "FREE" in u) or "THYROXINE FREE" in u or "THYROXINE.FREE" in u:
        return "3024-7"
    if "IRON SAT" in u or "IRON SATURATION" in u:
        return "2502-3"
    if "IRON BINDING" in u or _w(u, "TIBC"):
        return "2500-7"
    if (_w(u, "CK") or "CREATINE KINASE" in u) and "MB" not in u and "CKMB" not in u:
        return "2157-6"
    if _w(u, "ESR") or "SEDIMENTATION" in u:
        return "4537-7"
    if _w(u, "FSH") or "FOLLITROPIN" in u or "FOLLICLE STIM" in u:
        return "15067-2"
    if _w(u, "LH") or "LUTROPIN" in u or "LUTEINIZING" in u:
        return "10501-5"
    if _w(u, "LDH") or "LACTATE DEHYDROGENASE" in u:
        return "14805-6"
    if _w(u, "GGT") or "GLUTAMYL" in u:
        return "2324-2"
    if "TROPONIN" in u and ("HIGH SENS" in u or "HIGH-SENS" in u or _w(u, "HS")):
        return "89579-7"
    if "TROPONIN" in u:
        return "10839-9"
    if "HCG" in u or "CHORIOGONADOTROPIN" in u:
        return "19080-1"
    if "PHOSPHORUS" in u or _w(u, "PHOSPHATE"):
        return "2777-1"
    if ("TOTAL PROTEIN" in u or ("PROTEIN" in u and "TOTAL" in u)) and "CSF" not in u:
        return "2885-2"
    if "URIC ACID" in u or _w(u, "URATE"):
        return "3084-1"
    if _w(u, "APTT", "PTT") and "LUPUS" not in u:
        return "14979-9"
    if _w(u, "INR"):                       # check INR before PT ("PT INR" is INR, not PT-seconds)
        return "6301-6"
    if (_w(u, "PT") or "PROTHROMBIN" in u) and "INR" not in u:
        return "5902-2"
    if "TESTOSTERONE" in u and "FREE" not in u and "BIOAVAIL" not in u:
        return "2986-8"
    if "D-DIMER" in u or "D DIMER" in u or "FIBRIN D" in u:
        return "48058-2"
    if "PARATHYROID HORMONE" in u or _w(u, "PTH"):
        return "2731-8"
    if (_w(u, "BNP") or "NATRIURETIC PEPTIDE" in u) and "N-TERMINAL" not in u and "NT-PRO" not in u and "PROBNP" not in u:
        return "30934-4"
    if "BILIRUBIN" in u and "DIRECT" in u:
        return "1968-7"
    if "GLUCOSE" in u and "FASTING" in u:
        return "1558-6"
    if "ESTIMATED AVERAGE GLUCOSE" in u or "EST AVG GLUC" in u:
        return "27353-2"
    if "TRIGLYCERIDE" in u:
        return "2571-8"
    if _w(u, "T3") and "FREE" not in u and "REVERSE" not in u and "UPTAKE" not in u:
        return "3053-6"
    if _w(u, "C3") or "COMPLEMENT C3" in u:
        return "4485-9"
    if _w(u, "C4") or "COMPLEMENT C4" in u:
        return "4498-2"
    if "HEPATITIS B" in u and "SURFACE" in u and ("AB" in u or "ANTIBODY" in u):
        return "5193-8"
    if "ELECTROPHORESIS" in u or "GLOBULIN" in u:
        if "ALBUMIN" in u:
            return "2862-1"
        if "ALPHA-1" in u or "ALPHA 1" in u or "ALPHA1" in u:
            return "2865-4"
        if "ALPHA-2" in u or "ALPHA 2" in u or "ALPHA2" in u:
            return "2868-8"
        if "BETA" in u:
            return "2871-2"
        if "GAMMA" in u:
            return "2874-6"
    return None
