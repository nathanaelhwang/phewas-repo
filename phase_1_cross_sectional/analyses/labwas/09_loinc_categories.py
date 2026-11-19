# %% [markdown]
# Step 09 - Assign each LOINC analyte one of 16 clinical categories (for LabWAS
# Manhattan coloring / phecode-category analog).
# Backbone = LOINC CLASS; the undifferentiated CHEM bucket (and a few overrides)
# resolved by a curated COMPONENT/name classifier. Output: loinc_category.csv.

# %%
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import duckdb
import pandas as pd

import labwas_utils as U

CATEGORIES = ["blood", "cancer", "cardiovascular", "coagulation", "endocrine", "immune",
              "infectious", "kidney", "liver", "metabolic", "nutrition/vitamins", "ob/gyn",
              "other", "toxicology/pharmacology", "urinary"]  # + allergy = 16

CLASS_MAP = {"HEM/BC": "blood", "DRUG/TOX": "toxicology/pharmacology", "UA": "urinary",
             "CHAL.ROUTINE": "metabolic", "ALLERGY": "allergy", "SERO": "immune",
             "MICRO": "infectious", "COAG": "coagulation"}


def categorize(cls: str, comp: str, system: str, name: str) -> str:
    c, nm, sy = str(comp).upper(), str(name).upper(), str(system)

    def has(*kw):
        return any(k in c or k in nm for k in kw)

    # urine chemistries -> urinary regardless of CLASS
    if sy == "Urine" or "IN URINE" in nm:
        return "urinary"
    # non-blood specimen cell counts (CSF / body fluid) -> other
    if sy in ("CSF", "Body fld") or "CEREBRAL SPINAL" in nm or "BODY FLUID" in nm:
        return "other"

    # ---- CLASS backbone (with a couple of within-class overrides) ----
    if cls in CLASS_MAP:
        if cls == "HEM/BC" and "COMPLEMENT" in c:            # C3/C4 are immune
            return "immune"
        if cls == "DRUG/TOX" and has("COPPER", "ZINC", "SELENIUM", "MANGANESE", "PYRIDOXAL"):
            return "nutrition/vitamins"
        return CLASS_MAP[cls]

    # ---- CHEM (and anything unclassed): curated component classifier (ordered) ----
    if has("PROCALCITONIN"):                                 # sepsis/inflammation, not 'calcitonin'
        return "immune"
    if has("HYDROXYVITAMIN D", "CALCIFEDIOL", "DIHYDROXYVITAMIN", "COBALAMIN", "FOLATE",
           "FERRITIN", "IRON", "TRANSFERRIN", "THIAMINE", "PYRIDOX", "TOCOPHEROL",
           "CAROTENE", "PHYTONADIONE", "NIACIN", "NICOTINAMIDE", "ASCORB", "VITAMIN"):
        return "nutrition/vitamins"
    if has("PROSTATE SPECIFIC", "FETOPROTEIN", "MICROGLOBULIN", "THYROGLOBULIN",
           "PROSTATE HEALTH", "CARCINOEMBRYONIC", "CANCER AG", "CA 125", "CA 19-9"):
        return "cancer"
    if has("TROPONIN", "NATRIURETIC"):
        return "cardiovascular"
    if has("CHORIOGONADOTROPIN", "PROGESTERONE", "INHIBIN", "17-HYDROXYPROGESTERONE"):
        return "ob/gyn"
    if has("SEX HORMONE BINDING", "THYROTROPIN", "THYROXINE", "TRIIODOTHYRONINE",
           "PARATHYRIN", "CORTISOL", "FOLLITROPIN", "LUTROPIN", "PROLACTIN",
           "TESTOSTERONE", "ESTRADIOL", "ALDOSTERONE", "RENIN", "CORTICOTROPIN",
           "INSULIN-LIKE GROWTH", "METANEPHRINE", "CHROMOGRANIN", "GASTRIN",
           "CALCITONIN", "ANDROSTENEDIONE", "GROWTH FACTOR"):
        return "endocrine"
    if has("IMMUNOGLOBULIN", "IGG", "IGA", "IGM", "IGE", " GLOBULIN", "COMPLEMENT",
           "C REACTIVE", "RHEUMATOID", "PEROXIDASE AB", "ACETYLCHOLINE RECEPTOR",
           "TRANSGLUTAMINASE", "CITRULLINATED", "RNA POLYMERASE", "MYELOPEROXIDASE",
           "INTERLEUKIN", "TRYPTASE", "CERULOPLASMIN"):
        return "immune"
    if has("CREATININE", "UREA NITROGEN", "GLOMERULAR FILTRATION", "CYSTATIN"):
        return "kidney"
    if has("ALANINE AMINOTRANS", "ASPARTATE AMINOTRANS", "ALKALINE PHOSPHATASE",
           "BILIRUBIN", "GLUTAMYL TRANSFERASE", "ALBUMIN", "PROTEIN", "AMMONIA",
           "5'-NUCLEOTIDASE"):
        return "liver"
    if has("GLUCOSE", "HEMOGLOBIN A1C", "CHOLESTEROL", "TRIGLYCERIDE", "LIPOPROTEIN",
           "APOLIPOPROTEIN", "SODIUM", "POTASSIUM", "CHLORIDE", "CARBON DIOXIDE",
           "BICARBONATE", "ANION GAP", "CALCIUM", "MAGNESIUM", "PHOSPHATE", "URATE",
           "LACTATE", "OSMOL", "FRUCTOSAMINE", "C PEPTIDE", "INSULIN", "HYDROXYBUTYRATE",
           "LIPASE", "AMYLASE", "URIC ACID"):
        return "metabolic"
    return "other"


def main() -> None:
    con = duckdb.connect()
    dd = pd.read_csv(U.RESULTS / "column_data_dictionary.csv")
    lp = (U.CROSSWALK_DIR / "loinc_table.parquet").as_posix()
    meta = con.sql(f"SELECT LOINC_NUM AS loinc_code, CLASS, COMPONENT, SYSTEM, LONG_COMMON_NAME "
                   f"FROM read_parquet('{lp}')").df()
    d = dd[["loinc_code", "loinc_long_common_name"]].merge(meta, on="loinc_code", how="left")
    d["category"] = [categorize(r.CLASS, r.COMPONENT, r.SYSTEM, r.LONG_COMMON_NAME) for r in d.itertuples(index=False)]

    out = d[["loinc_code", "loinc_long_common_name", "CLASS", "SYSTEM", "category"]]
    out.to_csv(U.RESULTS / "loinc_category.csv", index=False)
    print(f"Wrote loinc_category.csv ({len(out)} analytes)")
    print("\n=== category counts ===")
    print(out["category"].value_counts().to_string())
    print("\n=== 'other' + spot-check assignments (review) ===")
    for r in out.sort_values("category").itertuples(index=False):
        print(f"  {r.category:<24} {r.loinc_code:<9} {str(r.CLASS):<12} {str(r.loinc_long_common_name)[:46]}")


if __name__ == "__main__":
    main()
