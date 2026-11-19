# %% [markdown]
# Step 09 - Attach human-readable labels + the super-heading category to each tested GPI-4.
#
# Category = the 17 Medi-Span super-headings (the Manhattan color axis / phecode-category
# analog). Names come from the Appendix-A crosswalk (step 03); GPI-4 classes the parse
# missed get a modal-GENNAME label from the fills (needs_review=True). Writes gpi_category.csv.

# %%
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import duckdb
import pandas as pd

import medwas_utils as U


def main() -> None:
    U.ensure_dirs()
    cw = pd.read_csv(U.CROSSWALK_DIR / "gpi_crosswalk.csv", dtype=str).set_index("gpi4")

    # union of tested classes across both windows
    tested = set()
    for label, _ in U.WINDOWS:
        tested |= set(pd.read_csv(U.ANALYSIS / f"kept_classes_{label}.csv", dtype=str)["gpi4"])
    tested = sorted(tested)

    # modal GENNAME per gpi4 (fallback label)
    con = duckdb.connect()
    U.configure_duckdb(con, __file__, memory_gb=48, threads=8)
    fills = (U.FILLS_DIR / "distinct_fills.parquet").as_posix()
    modal = con.sql(f"""
        SELECT gpi4, arg_max(genname, cnt) AS genname FROM (
            SELECT gpi4, genname, count(*) cnt FROM read_parquet('{fills}')
            WHERE genname IS NOT NULL AND gpi4 <> 'UNCL' GROUP BY gpi4, genname
        ) GROUP BY gpi4
    """).df().set_index("gpi4")["genname"].to_dict()

    rows = []
    for g in tested:
        gpi2 = g[:2]
        if g in cw.index:
            rows.append(dict(code=g, drug_class_name=cw.at[g, "drug_class_name"], gpi2=gpi2,
                             drug_group_name=cw.at[g, "drug_group_name"],
                             category=U.super_heading(gpi2),
                             label_source="appendix_a", needs_review=False))
        else:
            rows.append(dict(code=g, drug_class_name=modal.get(g, ""), gpi2=gpi2,
                             drug_group_name="", category=U.super_heading(gpi2),
                             label_source="genname_modal", needs_review=True))
    out = pd.DataFrame(rows)
    outp = U.RESULTS / "gpi_category.csv"
    out.to_csv(outp, index=False)

    n_appx = int((out["label_source"] == "appendix_a").sum())
    print(f"tested GPI-4 classes: {len(out)}")
    print(f"  labeled from Appendix A: {n_appx} ({100*n_appx/max(len(out),1):.0f}%)")
    print(f"  labeled from modal GENNAME (needs_review): {len(out)-n_appx}")
    print("category (super-heading) coverage:")
    for cat, n in out["category"].value_counts().items():
        print(f"  {cat:26s} {n}")
    print(f"\nWrote {outp}")
    U.log_milestone(f"Step 09 categories: {len(out)} tested GPI-4 classes labeled.")


if __name__ == "__main__":
    main()
