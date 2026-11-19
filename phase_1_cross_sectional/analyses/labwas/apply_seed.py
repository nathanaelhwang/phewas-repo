# %% [markdown]
# Apply verified clinical + allergen LOINC seeds to the crosswalk DRAFT.
# Fills chosen_loinc for currently-unmapped descriptors with n_mrn >= MIN_NMRN,
# using clinical_seed.classify (verified clinical codes) and an allergen matcher
# (manual fixes + fuzzy>=85 against the LOINC IgE pool + total-IgE special case).
# Persisted in chosen_loinc so it survives future 04 re-runs. Re-run 04->05->06->07 after.

# %%
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import duckdb
import pandas as pd
from rapidfuzz import fuzz, process

import labwas_utils as U
from clinical_seed import classify

MIN_NMRN = 1000

# Verified single-allergen overrides where fuzzy mismatches (checked in order).
ALLERGEN_SEED = {
    "RUSSIAN OLIVE": "13177-1", "OLIVE": "112110-2",
    "RYEGRASS": "112116-9", "RYE GRASS": "112116-9",
    "SALTWORT": "112096-3", "RUSSIAN THISTLE": "112096-3",
    "MUGWORT": "112149-0", "ELM": "112173-0", "SOY": "112121-9",
    "WHITE ASH": "6278-6", "TIMOTHY": "112105-2",
}
TOTAL_IGE = "19113-0"  # IgE [Units/volume] in Serum or Plasma


def _clean_allergen(u: str) -> str:
    s = re.sub(r"\bIGE\b|\bCLASS\b|\bQN\b|\bSER\b|\bAB\b|\bRAST\b|\bSERUM\b|\bMG/L\b", " ", u)
    s = re.sub(r"[^A-Z0-9 ]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def main() -> None:
    con = duckdb.connect()
    lp = (U.CROSSWALK_DIR / "loinc_table.parquet").as_posix()
    pool = con.sql(f"""SELECT LOINC_NUM, upper(LONG_COMMON_NAME) AS lcn
        FROM read_parquet('{lp}')
        WHERE STATUS='ACTIVE' AND CLASSTYPE='1' AND COMPONENT ILIKE '%Ab.IgE%'
          AND SCALE_TYP IN ('Qn','Ord') AND SYSTEM IN ('Ser','Ser/Plas','Plas')""").df()
    choices = pool["lcn"].tolist()

    def match_allergen(desc: str):
        u = desc.upper()
        if re.sub(r"[^A-Z]", "", u) in ("IGE", "IGEQN", "IGETOTAL", "TOTALIGE"):
            return TOTAL_IGE
        for kw, code in ALLERGEN_SEED.items():
            if kw in u:
                return code
        hit = process.extractOne(_clean_allergen(u), choices, scorer=fuzz.token_set_ratio)
        if hit and hit[1] >= 85:
            return pool.iloc[hit[2]]["LOINC_NUM"]
        return None

    draft = pd.read_csv(U.CROSSWALK_DIR / "loinc_crosswalk_DRAFT.csv", dtype=str).fillna("")
    nmrn = dict(zip(pd.read_csv(U.CROSSWALK_DIR / "lab_descriptors.csv")["descriptor"].astype(str),
                    pd.read_csv(U.CROSSWALK_DIR / "lab_descriptors.csv")["n_mrn"]))

    n_clin = n_alg = 0
    for i, row in draft.iterrows():
        d = str(row["descriptor"])
        if (row["chosen_loinc"].strip() or row["auto_loinc"].strip()):
            continue  # already mapped (auto or earlier manual) -> leave
        if nmrn.get(d, 0) < MIN_NMRN:
            continue
        if re.search(r"\bIGE\b", d, re.I):
            code = match_allergen(d)
            method = "allergen_seed"
            if code:
                n_alg += 1
        else:
            code = classify(d)
            method = "clinical_seed"
            if code:
                n_clin += 1
        if code:
            draft.at[i, "chosen_loinc"] = code
            draft.at[i, "mapping_method"] = method
            draft.at[i, "needs_review"] = "False"
            draft.at[i, "notes"] = (str(row["notes"]).strip() + " seed-assigned").strip()

    draft.to_csv(U.CROSSWALK_DIR / "loinc_crosswalk_DRAFT.csv", index=False)
    print(f"clinical seed assigned: {n_clin}")
    print(f"allergen seed assigned: {n_alg}")
    print(f"total newly assigned:   {n_clin + n_alg}")
    U.log_milestone(f"Seed apply: +{n_clin} clinical, +{n_alg} allergen LOINC mappings (n_mrn>={MIN_NMRN}).")


if __name__ == "__main__":
    main()
