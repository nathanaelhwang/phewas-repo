# %% [markdown]
# Step 4 - Finalize the crosswalk from the (optionally human-reviewed) DRAFT.
#
# For each descriptor: final code = chosen_loinc if the reviewer filled it, else
# auto_loinc. Re-joins full LOINC attributes, validates every code exists and is
# ACTIVE, and writes crosswalk/loinc_crosswalk.csv + loinc_table_manifest.json.
# Runs end-to-end even with an un-reviewed DRAFT (auto values flow through).

# %%
import hashlib
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import duckdb
import pandas as pd

import labwas_utils as U

FINAL_COLS = ["lab", "loinc_code", "loinc_long_common_name", "loinc_component",
              "loinc_system", "loinc_scale", "mapping_method", "confidence",
              "needs_review", "notes"]


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    draft_path = U.CROSSWALK_DIR / "loinc_crosswalk_DRAFT.csv"
    lp = (U.CROSSWALK_DIR / "loinc_table.parquet").as_posix()
    out = U.CROSSWALK_DIR / "loinc_crosswalk.csv"
    manifest = U.CROSSWALK_DIR / "loinc_table_manifest.json"

    draft = pd.read_csv(draft_path, dtype=str).fillna("")
    con = duckdb.connect()
    attrs = con.sql(f"""
        SELECT LOINC_NUM AS loinc_code, LONG_COMMON_NAME AS loinc_long_common_name,
               COMPONENT AS loinc_component, SYSTEM AS loinc_system,
               SCALE_TYP AS loinc_scale, STATUS
        FROM read_parquet('{lp}')
    """).df()
    active = dict(zip(attrs["loinc_code"], attrs["STATUS"]))
    attr_map = attrs.drop(columns="STATUS").set_index("loinc_code")

    rows, n_manual, n_invalid = [], 0, 0
    for d in draft.itertuples(index=False):
        chosen = str(d.chosen_loinc).strip()
        code = chosen if chosen else str(d.auto_loinc).strip()
        method, conf, notes = d.mapping_method, d.confidence, str(d.notes)
        if chosen:
            method, n_manual = "manual", n_manual + 1

        rec = {c: "" for c in FINAL_COLS}
        rec["lab"] = d.descriptor
        rec["mapping_method"] = method
        rec["confidence"] = conf
        rec["notes"] = notes

        if code and code in attr_map.index:
            if active.get(code) != "ACTIVE":
                notes = (notes + "; " if notes else "") + f"WARNING: LOINC {code} not ACTIVE"
            a = attr_map.loc[code]
            rec.update({"loinc_code": code,
                        "loinc_long_common_name": a["loinc_long_common_name"],
                        "loinc_component": a["loinc_component"],
                        "loinc_system": a["loinc_system"],
                        "loinc_scale": a["loinc_scale"],
                        "needs_review": False, "notes": notes})
        elif code:  # a code was supplied but is not in the table -> invalid, do not use
            n_invalid += 1
            rec.update({"loinc_code": "", "needs_review": True,
                        "notes": (notes + "; " if notes else "") + f"INVALID code '{code}' not in LOINC table"})
        else:  # no code -> stays unmapped / needs review
            rec.update({"loinc_code": "", "needs_review": True})
        rows.append(rec)

    final = pd.DataFrame(rows, columns=FINAL_COLS)
    final.to_csv(out, index=False)

    ver = re.search(r"Loinc_([\d.]+)", U.LOINC_CSV)
    manifest.write_text(json.dumps({
        "loinc_version": ver.group(1) if ver else "unknown",
        "source_path": U.LOINC_CSV,
        "sha256": _sha256(U.LOINC_CSV),
        "n_codes": int(len(attrs)),
    }, indent=2), encoding="utf-8")

    n = len(final)
    n_mapped = int((final["loinc_code"] != "").sum())
    print("\n==== Step 4 finalize crosswalk ====")
    print(f"descriptors:            {n:,}")
    print(f"mapped to a LOINC code: {n_mapped:,} ({100 * n_mapped / n:.1f}%)")
    print(f"manual overrides:       {n_manual:,}")
    print(f"invalid codes dropped:  {n_invalid:,}")
    print(f"still needs_review:     {int(final['needs_review'].sum()):,}")
    print(f"Wrote {out}")
    print(f"Wrote {manifest}")
    U.log_milestone(f"Step 4 finalize: {n_mapped:,}/{n:,} descriptors mapped "
                    f"({n_manual:,} manual; {n_invalid:,} invalid).")


if __name__ == "__main__":
    main()
