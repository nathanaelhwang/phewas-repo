# %% [markdown]
# Step 03 - Parse Medi-Span TCS Appendix A into a GPI crosswalk.
#
# Each well-formed line is "GG CC SS   *Name*(1-3 asterisks)", where the trailing-asterisk
# count encodes the level: 1 = Drug Group (gpi2), 2 = Drug Class (gpi4), 3 = Drug Subclass
# (gpi6). We build gpi2->group_name and gpi4->class_name; super_heading comes purely from
# the fixed 2-digit range table (medwas_utils.super_heading), so it is populated even where
# the OCR text is garbled. Writes crosswalk/gpi_crosswalk.csv (one row per gpi4 in the doc).

# %%
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pandas as pd

import medwas_utils as U

ROW = re.compile(r"^\s*(\d{2})\s+(\d{2})\s+(\d{2})\s+\*(.+?)(\*{1,3})\s*$")


def main() -> None:
    U.ensure_dirs()
    text = U.GPI_DOC.read_text(encoding="utf-8", errors="replace").splitlines()

    group_name: dict[str, str] = {}   # gpi2 -> DRUG GROUP name
    class_name: dict[str, str] = {}   # gpi4 -> Drug Class name
    n_sub = 0
    for line in text:
        m = ROW.match(line)
        if not m:
            continue
        g, c, s, name, stars = m.group(1), m.group(2), m.group(3), m.group(4).strip(), m.group(5)
        level = len(stars)
        if level == 1 and c == "00" and s == "00":
            group_name[g] = name
        elif level == 2 and s == "00":               # includes combination classes (CC=99)
            class_name[g + c] = name
        elif level == 3:
            n_sub += 1                                 # gpi6 subclasses: counted, not kept

    # Some Drug Groups have no sub-classes, so drugs sit at the group-level code
    # GG00 (e.g. 34 00 = Calcium Blockers). Give those the GROUP name as their class name.
    for gpi2, gname in group_name.items():
        class_name.setdefault(gpi2 + "00", gname)

    rows = []
    for gpi4 in sorted(class_name):
        gpi2 = gpi4[:2]
        rows.append({
            "gpi4": gpi4,
            "drug_class_name": class_name[gpi4],
            "gpi2": gpi2,
            "drug_group_name": group_name.get(gpi2, ""),
            "super_heading": U.super_heading(gpi2),
            "label_source": "appendix_a",
            "needs_review": group_name.get(gpi2, "") == "",
        })
    cw = pd.DataFrame(rows)
    out = U.CROSSWALK_DIR / "gpi_crosswalk.csv"
    cw.to_csv(out, index=False)

    print("==== Step 03 GPI crosswalk (from Appendix A) ====")
    print(f"drug groups (gpi2) parsed:   {len(group_name)}")
    print(f"drug classes (gpi4) parsed:  {len(class_name)}")
    print(f"subclasses (gpi6) seen:      {n_sub}")
    print(f"gpi4 rows missing group name: {int(cw['needs_review'].sum())}")
    print("\nsuper-heading coverage (gpi4 count):")
    for sh, n in cw["super_heading"].value_counts().items():
        print(f"  {sh:26s} {n}")
    print(f"\nWrote {out}")
    U.log_milestone(f"Step 03 crosswalk: {len(class_name)} GPI-4 classes, "
                    f"{len(group_name)} GPI-2 groups parsed from Appendix A.")


if __name__ == "__main__":
    main()
