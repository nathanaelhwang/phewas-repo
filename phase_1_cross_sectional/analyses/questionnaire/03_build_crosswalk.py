# %% [markdown]
# Step 3 - Build the question -> target-code crosswalk for HUMAN REVIEW.
#
# KPHC side is mechanical & authoritative (QUEST_ID -> col-I code, exact decode
# maps verified in 02). Somnoware side is PROPOSED by text anchors and must be
# reviewed. Judgment-call decode rules are flagged needs_review=True.
#
# Emits:
#   results/questionnaire_results/crosswalk.csv               (review this)
#   results/questionnaire_results/somnoware_unmatched_questions.csv
#   results/questionnaire_results/crosswalk_decode_rules.md
#
# ==> REVIEW crosswalk.csv, correct it, save as `crosswalk_reviewed.csv`.
#     Steps 04/05 read crosswalk_reviewed.csv if present, else crosswalk.csv.

# %%
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import questionnaire_utils as U

DECODE_SUMMARY = {
    "ess": "0-3 doze likelihood (No/Slight/Moderate/High; Somnoware '0 (..)'-prefixed)",
    "freq": "0-4 (Never very rarely/1-2 per mo/1-2 per wk/3-4 per wk/Almost every day)",
    "yesno": "Yes->1, No->0 (ambiguous->NA)",
    "isi_sev": "0-4 (None/Mild/Moderate/Severe/Very severe)",
    "isi_sat": "0-4 REVERSE (Very satisfied->0 ... Very dissatisfied->4)",
    "isi_imp": "0-4 (Not at all->0 / A little / Somewhat / Much / Very much->4)",
    "phq": "0-3 (Not at all/Several days/More than half/Nearly every day)",
    "fosq": "0-3 (No->0 / Yes a little / moderate / extreme->3); 'other reasons'->NA",
    "time": "minutes since midnight from HH:MM (late-night bedtimes wrap)",
    "latency": "categorical bins -> midpoint minutes (<15->8 ... >4h->270)",
    "count": "integer count; 'Never'->0, '>8'->9",
    "neck": "STOP-BANG neck: collar >40cm(15.75in) -> 1 else 0",
    "hours": "hours of sleep per night, integer 0-24",
    "duration": "nap duration bins -> midpoint minutes",
    "text": "raw category text (chief complaint), no numeric decode",
    "empty": "EHR-derived (STOP5/6/8) -> left blank",
}

# KPHC codes whose decode rule is a judgment call -> force review
KPHC_REVIEW_NOTES = {
    "ISI4": "satisfaction REVERSE-scored (Very Satisfied=0 .. Very Dissatisfied=4) - confirm direction",
    "SVS1": "answers are 0-7 counts, not Yes/No despite col-I phrasing - confirm count semantics",
    "SVS2": "answers are 0-7 counts, not Yes/No despite col-I phrasing - confirm count semantics",
    "NAPS": "answers are 0-7 days/week count, not Yes/No - confirm",
    "LATENCY": "categorical bins mapped to MIDPOINT minutes - confirm (vs ordinal)",
    "BEDTIME": "minutes since midnight; late-night values wrap (23:00=1380, 00:30=30) - confirm",
    "WAKETIME": "minutes since midnight - confirm",
    **{f"FOSQ{i}": "encoded No=0..extreme=3 ('other reasons'->NA); standard FOSQ reverses - confirm direction"
       for i in range(1, 11)},
}


def kphc_rows() -> list[dict]:
    qmap = U.load_kphc_quest_map()
    qmap = qmap[qmap["code"].isin(U.TARGET_CODES)]
    prof = json.loads((U.PROFILE_DIR / "kphc_profile.json").read_text())["per_code"]
    rows = []
    for code in U.TARGET_CODES:
        fam = U.FAMILY[code]
        hit = qmap[qmap["code"] == code]
        if hit.empty:                 # STOP5/6/8 (empty), NAPDURATION (Somnoware-only)
            continue
        rec = hit.iloc[0]
        pc = prof.get(code, {})
        ex = "; ".join(f"{a!r}" for a, _ in pc.get("answer_options", [])[:5])
        rows.append({
            "target_code": code, "item_family": fam, "source": "kphc",
            "source_form": "GENERAL INTAKE SLEEP HISTORY PEQ SCAL",
            "match_key": rec["quest_id"], "question_text": rec["question"],
            "decode_summary": DECODE_SUMMARY[fam], "example_answers": ex,
            "n_encounters": pc.get("n_encounters", 0),
            "confidence": "high",
            "needs_review": code in KPHC_REVIEW_NOTES,
            "notes": KPHC_REVIEW_NOTES.get(code, ""),
        })
    return rows


def somnoware_rows() -> tuple[list[dict], list[dict]]:
    prof = json.loads((U.PROFILE_DIR / "somnoware_profile.json").read_text())
    matched, unmatched = [], []
    for form, d in prof["in_scope_forms"].items():
        if form.casefold() in U.SOMNO_FORM_DENYLIST:   # admin/PAP/consent forms
            continue
        for q in d["questions"]:
            qt = q["question"]
            hits = U.match_question(qt)
            opts = "; ".join(repr(o) for o in (q.get("option_text_samples") or [])[:5])
            if len(hits) == 1:
                code = hits[0]
                matched.append({
                    "target_code": code, "item_family": U.FAMILY[code],
                    "source": "somnoware", "source_form": form,
                    "match_key": qt[:120], "question_text": qt,
                    "decode_summary": DECODE_SUMMARY[U.FAMILY[code]],
                    "example_answers": opts, "n_encounters": q["n"],
                    "confidence": "medium", "needs_review": True,
                    "notes": "text-anchor match - verify",
                })
            elif len(hits) > 1:
                matched.append({
                    "target_code": "", "item_family": "", "source": "somnoware",
                    "source_form": form, "match_key": qt[:120], "question_text": qt,
                    "decode_summary": "", "example_answers": opts, "n_encounters": q["n"],
                    "confidence": "conflict", "needs_review": True,
                    "notes": f"AMBIGUOUS - candidates: {', '.join(hits)}",
                })
            else:
                unmatched.append({"source_form": form, "question_text": qt,
                                  "n": q["n"], "option_text_samples": opts})
    return matched, unmatched


def write_decode_rules() -> None:
    L = ["# Decode rules (per item family)", "",
         "Decoded numeric values; nullable so NOT_ASKED / blank / unmapped stay <NA>.",
         "Handles both KPHC word labels and Somnoware digit-prefixed labels.", ""]
    for fam, summ in DECODE_SUMMARY.items():
        codes = [c for c in U.TARGET_CODES if U.FAMILY[c] == fam]
        lo, hi = U.FAMILY_RANGE[fam]
        L.append(f"- **{fam}** (range {lo}-{hi}) - {summ}")
        L.append(f"    - codes: {', '.join(codes)}")
    (U.RESULTS / "crosswalk_decode_rules.md").write_text("\n".join(L), encoding="utf-8")


def feedback_rows() -> list[dict]:
    """Reviewer FEEDBACK mappings + deferred/active matrix stems -> crosswalk rows."""
    fb = U.load_feedback()
    out = []
    for qn, code in fb["q2code"].items():
        out.append({"target_code": code, "item_family": U.FAMILY[code],
                    "source": "somnoware", "source_form": "(reviewer FEEDBACK)",
                    "match_key": qn[:120], "question_text": qn,
                    "decode_summary": DECODE_SUMMARY.get(U.FAMILY[code], ""),
                    "example_answers": "", "n_encounters": "", "confidence": "reviewer",
                    "needs_review": False, "notes": "user FEEDBACK mapping"})
    for si, spec in enumerate(U.MATRIX_SPECS):
        codes = spec["codes"]
        out.append({"target_code": ",".join(codes) if codes else "(deferred)",
                    "item_family": "matrix", "source": "somnoware_matrix",
                    "source_form": spec["form"], "match_key": spec["stem"],
                    "question_text": spec["stem"], "decode_summary": spec["note"],
                    "example_answers": "", "n_encounters": "",
                    "confidence": "matrix", "needs_review": codes is None,
                    "notes": "positional reconstruction" if codes else "deferred"})
    return out


def main() -> None:
    U.ensure_dirs()
    rows = kphc_rows()
    smatched, unmatched = somnoware_rows()
    cw = pd.DataFrame(rows + smatched + feedback_rows())
    cols = ["target_code", "item_family", "source", "source_form", "match_key",
            "question_text", "decode_summary", "example_answers", "n_encounters",
            "confidence", "needs_review", "notes"]
    cw = cw[cols]
    U.safe_to_csv(cw, U.RESULTS / "crosswalk.csv")
    U.safe_to_csv(pd.DataFrame(unmatched), U.RESULTS / "somnoware_unmatched_questions.csv")
    write_decode_rules()

    # coverage report
    som_codes = {r["target_code"] for r in smatched if r["target_code"]}
    only_kphc = [c for c in U.TARGET_CODES if c not in som_codes]
    conflicts = sum(1 for r in smatched if r["confidence"] == "conflict")
    print(f"[crosswalk] {len(rows)} KPHC rows (all 51 codes) + "
          f"{len(smatched)} Somnoware proposed rows ({conflicts} conflicts)")
    print(f"[crosswalk] Somnoware unmatched questions: {len(unmatched)}")
    print(f"[crosswalk] codes with a Somnoware source: {len(som_codes)}/51")
    print(f"[crosswalk] codes KPHC-only (no Somnoware match): {only_kphc}")
    print(f"[crosswalk] rows flagged needs_review: {int(cw['needs_review'].sum())}")
    print("\n==> REVIEW results/questionnaire_results/crosswalk.csv")
    print("    (+ somnoware_unmatched_questions.csv, crosswalk_decode_rules.md)")
    print("    correct it, save as crosswalk_reviewed.csv, then run 04/05.")
    U.log_milestone(
        f"03_build_crosswalk: {len(rows)} KPHC + {len(smatched)} Somnoware rows "
        f"({conflicts} conflicts, {len(unmatched)} unmatched); "
        f"{len(som_codes)}/51 codes have a Somnoware source. Awaiting review.")


if __name__ == "__main__":
    main()
