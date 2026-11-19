# %% [markdown]
# Step 3 - Build the descriptor -> LOINC DRAFT crosswalk (deterministic).
#
# 1. Convert Loinc.csv -> crosswalk/loinc_table.parquet (once).
# 2. Restrict the candidate pool to lab-orderable codes (ACTIVE, CLASSTYPE=1
#    Laboratory, SCALE_TYP in {Qn,Ord}).
# 3. Normalize + abbreviation-expand each descriptor, then match:
#    a) EXACT on normalized COMPONENT/SHORTNAME -> the canonical analyte family,
#       tie-broken by COMMON_TEST_RANK (most-ordered) then SYSTEM then scale.
#       This is what surfaces 718-7/2345-7/2160-0 etc. -- token_set_ratio alone
#       gives 100 to every component that merely *contains* the analyte token.
#    b) FUZZY token_sort_ratio fallback only when there is no exact match.
# 4. Assign confidence tiers; emit loinc_crosswalk_DRAFT.csv with top-k
#    candidates + auto_loinc (high/medium) + a blank chosen_loinc for review.
# De-identified (lab names + counts only).

# %%
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import duckdb
import pandas as pd
from rapidfuzz import fuzz, process

import labwas_utils as U

K = 5  # candidates surfaced per descriptor

# Whole-token abbreviation expansion for the common CBC/BMP/CMP/lipid/thyroid/
# coag analytes (the high-n_mrn core). Genuinely ambiguous tokens are expanded
# for candidate generation but force needs_review (see AMBIGUOUS).
ABBREV = {
    # CBC
    "HGB": "HEMOGLOBIN", "HB": "HEMOGLOBIN", "HCT": "HEMATOCRIT",
    "WBC": "LEUKOCYTES", "RBC": "ERYTHROCYTES", "PLT": "PLATELETS",
    "MCV": "ERYTHROCYTE MEAN CORPUSCULAR VOLUME",
    "MCH": "ERYTHROCYTE MEAN CORPUSCULAR HEMOGLOBIN",
    "MCHC": "ERYTHROCYTE MEAN CORPUSCULAR HEMOGLOBIN CONCENTRATION",
    "RDW": "ERYTHROCYTE DISTRIBUTION WIDTH", "MPV": "PLATELET MEAN VOLUME",
    # BMP / CMP
    "NA": "SODIUM", "K": "POTASSIUM", "CL": "CHLORIDE",
    "CO2": "CARBON DIOXIDE", "HCO3": "BICARBONATE", "BICARB": "BICARBONATE",
    "BUN": "UREA NITROGEN", "CR": "CREATININE", "CREAT": "CREATININE",
    "GLU": "GLUCOSE", "CA": "CALCIUM", "MG": "MAGNESIUM", "PHOS": "PHOSPHATE",
    "EGFR": "GLOMERULAR FILTRATION RATE",
    "ALB": "ALBUMIN", "TP": "PROTEIN", "TPROT": "PROTEIN",
    "TBILI": "BILIRUBIN TOTAL", "DBILI": "BILIRUBIN DIRECT",
    "ALT": "ALANINE AMINOTRANSFERASE", "SGPT": "ALANINE AMINOTRANSFERASE",
    "AST": "ASPARTATE AMINOTRANSFERASE", "SGOT": "ASPARTATE AMINOTRANSFERASE",
    "ALP": "ALKALINE PHOSPHATASE", "ALKPHOS": "ALKALINE PHOSPHATASE",
    # lipids / endocrine
    "CHOL": "CHOLESTEROL", "HDL": "CHOLESTEROL IN HDL", "LDL": "CHOLESTEROL IN LDL",
    "TG": "TRIGLYCERIDE", "TRIG": "TRIGLYCERIDE",
    "A1C": "HEMOGLOBIN A1C", "HBA1C": "HEMOGLOBIN A1C", "HGBA1C": "HEMOGLOBIN A1C",
    "TSH": "THYROTROPIN", "FT4": "THYROXINE FREE", "FT3": "TRIIODOTHYRONINE FREE",
    # coag / inflammation
    "INR": "INR", "PT": "PROTHROMBIN TIME", "PTT": "APTT", "APTT": "APTT",
    "ESR": "ERYTHROCYTE SEDIMENTATION RATE", "CRP": "C REACTIVE PROTEIN",
}
AMBIGUOUS = {"UA", "PT", "CA", "MG"}  # always needs_review even with a dict hit

# Method/specimen/scale qualifier tokens that LOINC encodes on non-COMPONENT
# axes (SYSTEM/METHOD/SCALE). Stripping them from the descriptor lets the base
# analyte exact-match a COMPONENT (e.g. "HCT, AUTO" -> "HCT" -> HEMATOCRIT).
NOISE = {"AUTO", "AUTOMATED", "COUNT", "MANUAL", "CALC", "CALCULATED", "CALCD",
         "EST", "ESTIMATE", "ESTIMATED", "BLOOD", "BLD", "SERUM", "SER",
         "PLASMA", "PLAS", "WHOLE", "RANDOM", "LEVEL", "POC", "SERPL"}

SYSTEM_PRIORITY = {
    "Bld": 0, "Bld/Ser/Plas": 1, "Ser/Plas": 1, "Plas/Ser": 1,
    "Ser": 2, "Plas": 2, "RBC": 3, "WBC": 3, "Urine": 5,
}

# Verified seed for high-volume CBC indices whose LOINC COMPONENT/SHORTNAME the
# fuzzy matcher cannot reach (LOINC models them as entitic/ratio terms). Keyed
# by the cleaned descriptor; each code checked ACTIVE + in-pool against v2.82.
MANUAL_MAP = {
    "HCT": "4544-3", "HEMATOCRIT": "4544-3",       # Hematocrit [VFr] of Blood
    "MCV": "787-2", "MCH": "785-6", "MCHC": "786-4",
}


def _norm(s: str) -> str:
    s = str(s).upper().strip()
    s = re.sub(r"\b([A-Z]+)'S\b", r"\1", s)  # possessive: WBC'S -> WBC
    s = re.sub(r"\(.*?\)", " ", s)        # drop parentheticals (often units)
    s = re.sub(r"[^A-Z0-9 ]+", " ", s)    # punctuation -> space
    return re.sub(r"\s+", " ", s).strip()


def _clean_desc(norm: str) -> str:
    """Drop method/specimen/scale qualifier tokens (NOISE) so the base analyte
    can exact-match a LOINC COMPONENT. Applied to descriptors only, not to the
    LOINC table side."""
    toks = [t for t in norm.split() if t not in NOISE]
    return " ".join(toks) if toks else norm


def _expand(norm: str) -> str:
    if norm in ABBREV:
        return ABBREV[norm]
    return " ".join(ABBREV.get(t, t) for t in norm.split())


def _rank_int(v) -> int:
    try:
        n = int(float(v))
        return n if n > 0 else 10**9
    except (ValueError, TypeError):
        return 10**9


def _prop_pri(p) -> int:
    """Prefer plain concentration properties over per-cell entitic indices
    (MCH/MCHC) and qualitative-ish properties, so a bare analyte name resolves
    to its concentration code (e.g. HEMOGLOBIN -> 718-7, not MCHC 786-4)."""
    p = str(p)
    if p.startswith("Ent") or p in ("PrThr", "Titr", "Naric", "Imp"):
        return 1
    return 0


def load_pool(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Convert Loinc.csv -> parquet (once) and return the filtered candidate pool."""
    lp = (U.CROSSWALK_DIR / "loinc_table.parquet").as_posix()
    if not Path(lp).exists():
        con.execute(f"""
            COPY (SELECT * FROM read_csv('{U.LOINC_CSV}', header=true, quote='"',
                                         escape='"', all_varchar=true, ignore_errors=true))
            TO '{lp}' (FORMAT PARQUET)
        """)
    pool = con.sql(f"""
        SELECT LOINC_NUM, COMPONENT, SHORTNAME, LONG_COMMON_NAME, SYSTEM,
               PROPERTY, SCALE_TYP, CLASS, COMMON_TEST_RANK
        FROM read_parquet('{lp}')
        WHERE STATUS = 'ACTIVE' AND CLASSTYPE = '1'
          AND SCALE_TYP IN ('Qn', 'Ord')
    """).df()
    pool["comp_norm"] = pool["COMPONENT"].map(_norm)
    pool["short_norm"] = pool["SHORTNAME"].map(_norm)
    pool["rank_int"] = pool["COMMON_TEST_RANK"].map(_rank_int)
    pool["sys_pri"] = pool["SYSTEM"].map(lambda s: SYSTEM_PRIORITY.get(s, 9))
    pool["scale_pri"] = (pool["SCALE_TYP"] != "Qn").astype(int)  # prefer quantitative
    pool["prop_pri"] = pool["PROPERTY"].map(_prop_pri)
    return pool


def build_index(pool: pd.DataFrame):
    comp_index: dict[str, list[int]] = defaultdict(list)
    short_index: dict[str, list[int]] = defaultdict(list)
    for i, v in enumerate(pool["comp_norm"].values):
        comp_index[v].append(i)
    for i, v in enumerate(pool["short_norm"].values):
        if v:
            short_index[v].append(i)
    cols = {
        "loinc": pool["LOINC_NUM"].to_numpy(),
        "lcn": pool["LONG_COMMON_NAME"].to_numpy(),
        "rank": pool["rank_int"].to_numpy(),
        "sys": pool["sys_pri"].to_numpy(),
        "scale": pool["scale_pri"].to_numpy(),
        "prop": pool["prop_pri"].to_numpy(),
        "comp_norm": pool["comp_norm"].tolist(),
    }
    return comp_index, short_index, cols


def _cand(cols, i: int, score: float) -> dict:
    return {"loinc": cols["loinc"][i], "lcn": cols["lcn"][i], "score": round(float(score), 1),
            "rank": int(cols["rank"][i]), "sys": int(cols["sys"][i]),
            "scale": int(cols["scale"][i]), "prop": int(cols["prop"][i])}


def match_one(cleaned: str, expanded: str, comp_index, short_index, cols) -> tuple[list[dict], bool]:
    """Return (top-K candidates, is_exact). Exact COMPONENT/SHORTNAME match wins
    (trying the expanded analyte name and the raw cleaned form), tie-broken by
    PROPERTY tier (concentration over entitic) then COMMON_TEST_RANK then SYSTEM
    then scale; else fuzzy fallback on expanded then cleaned."""
    for key, idx_map in [(expanded, comp_index), (cleaned, comp_index),
                         (expanded, short_index), (cleaned, short_index)]:
        idxs = idx_map.get(key)
        if idxs:
            rows = [_cand(cols, i, 100.0) for i in idxs]
            rows.sort(key=lambda d: (d["prop"], d["rank"], d["sys"], d["scale"], d["loinc"]))
            return rows[:K], True
    for q in (expanded, cleaned):
        hits = process.extract(q, cols["comp_norm"], scorer=fuzz.token_sort_ratio,
                               limit=K * 8, score_cutoff=70)
        if hits:
            rows = [_cand(cols, idx, score) for _s, score, idx in hits]
            rows.sort(key=lambda d: (-d["score"], d["prop"], d["rank"], d["sys"], d["scale"], d["loinc"]))
            return rows[:K], False
    return [], False


def main() -> None:
    U.ensure_dirs()
    desc_path = (U.CROSSWALK_DIR / "lab_descriptors.csv").as_posix()
    out = U.CROSSWALK_DIR / "loinc_crosswalk_DRAFT.csv"

    con = duckdb.connect()
    con.execute("PRAGMA threads=8")
    pool = load_pool(con)
    comp_index, short_index, cols = build_index(pool)
    code2idx = {c: i for i, c in enumerate(cols["loinc"])}
    print(f"LOINC candidate pool (ACTIVE, lab, Qn/Ord): {len(pool):,} codes", flush=True)

    desc = pd.read_csv(desc_path, dtype={"descriptor": str})
    print(f"descriptors to map: {len(desc):,}", flush=True)

    out_rows = []
    for i, d in enumerate(desc.itertuples(index=False)):
        verbatim = d.descriptor
        norm = _norm(verbatim)
        cleaned = _clean_desc(norm)
        expanded = _expand(cleaned)
        ambiguous = cleaned in AMBIGUOUS
        # a raw "%" marks a fraction/differential; after normalization it would
        # silently collapse onto the absolute-count code, so force review.
        is_pct = "%" in str(verbatim)

        if not is_pct and cleaned in MANUAL_MAP and MANUAL_MAP[cleaned] in code2idx:
            cands = [_cand(cols, code2idx[MANUAL_MAP[cleaned]], 100.0)]
            conf, method, review = "high", "manual_seed", False
        else:
            cands, is_exact = match_one(cleaned, expanded, comp_index, short_index, cols) if expanded else ([], False)
            top = cands[0] if cands else None
            second = cands[1] if len(cands) > 1 else None
            margin = (top["score"] - second["score"]) if (top and second) else (top["score"] if top else 0)
            if not top:
                conf, method, review = "low", "no_match", True
            elif ambiguous or is_pct:
                conf, method, review = "low", ("ambiguous" if ambiguous else "fraction_review"), True
            elif is_exact:
                conf, method, review = "high", ("abbrev_dict" if cleaned in ABBREV else "exact"), False
            elif top["score"] >= 92 and margin >= 8:
                conf, method, review = "medium", "fuzzy", False
            else:
                conf, method, review = "low", "fuzzy_lowconf", True

        top = cands[0] if cands else None

        auto_loinc = top["loinc"] if (top and conf in ("high", "medium")) else ""
        auto_lcn = top["lcn"] if (top and conf in ("high", "medium")) else ""

        row = {"descriptor": verbatim, "n_mrn": d.n_mrn, "n_rows": d.n_rows}
        for k in range(K):
            c = cands[k] if k < len(cands) else None
            row[f"cand{k+1}_loinc"] = c["loinc"] if c else ""
            row[f"cand{k+1}_lcn"] = c["lcn"] if c else ""
            row[f"cand{k+1}_score"] = c["score"] if c else ""
            row[f"cand{k+1}_rank"] = c["rank"] if c else ""
        row.update({"auto_loinc": auto_loinc, "auto_lcn": auto_lcn,
                    "chosen_loinc": "", "mapping_method": method,
                    "confidence": conf, "needs_review": review, "notes": ""})
        out_rows.append(row)
        if (i + 1) % 250 == 0:
            print(f"  matched {i + 1:,}/{len(desc):,}", flush=True)

    draft = pd.DataFrame(out_rows).sort_values("n_mrn", ascending=False)
    draft.to_csv(out, index=False)

    n = len(draft)
    n_auto = int((draft["auto_loinc"] != "").sum())
    n_review = int(draft["needs_review"].sum())
    mrn_total = draft["n_mrn"].sum()
    mrn_auto = draft.loc[draft["auto_loinc"] != "", "n_mrn"].sum()
    print("\n==== Step 3 crosswalk DRAFT ====")
    print(f"descriptors: {n:,}")
    print(f"auto-mapped (high/medium): {n_auto:,} ({100 * n_auto / n:.1f}%)")
    print(f"needs_review: {n_review:,}")
    print(f"n_mrn-weighted auto coverage: {100 * mrn_auto / mrn_total:.1f}%")
    print(f"Wrote {out}")
    U.log_milestone(
        f"Step 3 crosswalk DRAFT: {n:,} descriptors; {n_auto:,} auto-mapped "
        f"({100 * mrn_auto / mrn_total:.1f}% n_mrn-weighted); {n_review:,} need review."
    )


if __name__ == "__main__":
    main()
