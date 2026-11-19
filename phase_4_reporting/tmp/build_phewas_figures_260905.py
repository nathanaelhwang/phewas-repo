from __future__ import annotations

import csv
import json
import math
import re
import textwrap
from collections import defaultdict
from pathlib import Path
from typing import Callable, Iterable

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(r"C:\Users\natha\Projects\work\phewas")
OUT = ROOT / "tmp" / "phewas_260905_assets"
OUT.mkdir(parents=True, exist_ok=True)

WIDTH = 1950  # 6.5 inches at 300 dpi
DPI = (300, 300)
Y_CAP = 50.0

BG = "#FBFCFA"
PANEL_BG = "#FFFFFF"
INK = "#152D32"
MID_INK = "#4C6165"
GRID = "#D9E2E0"
TEAL = "#007C76"
TEAL_LIGHT = "#9BCBC7"
PURPLE = "#6B4C9A"
PURPLE_LIGHT = "#C0B2D4"
OUTLINE = "#173C3D"
MUTED = "#A7B3B2"
BAND_A = "#EDF5F3"
BAND_B = "#F6F3F8"
BAND_STRONG = "#DCECE8"
WHITE = "#FFFFFF"

FONT_SANS = Path(r"C:\Windows\Fonts\arial.ttf")
FONT_SANS_BOLD = Path(r"C:\Windows\Fonts\arialbd.ttf")
FONT_SANS_ITALIC = Path(r"C:\Windows\Fonts\ariali.ttf")
FONT_SERIF = Path(r"C:\Windows\Fonts\georgia.ttf")
FONT_SERIF_BOLD = Path(r"C:\Windows\Fonts\georgiab.ttf")

SAFE_INPUTS = {
    "results/medwas_results/analysis/results_1yr.csv",
    "results/medwas_results/analysis/results_5yr.csv",
    "results/medwas_results/gpi_category.csv",
    "results/procwas_results/analysis/results_1yr.csv",
    "results/procwas_results/analysis/results_5yr.csv",
    "results/utilwas_results/analysis/results_phecode_1yr.csv",
    "results/utilwas_results/analysis/results_phecode_5yr.csv",
    "results/qwas_results/analysis/results.csv",
    "results/questionnaire_results/crosswalk.csv",
    "results/behwas_results/analysis/results_all.csv",
}

FORBIDDEN_PARTS = ("data/raw", "/phi/", "patient_records", "_identified", "/.ssh/", "/.aws/")


def font(size: int, *, bold: bool = False, serif: bool = False, italic: bool = False) -> ImageFont.FreeTypeFont:
    if serif:
        path = FONT_SERIF_BOLD if bold else FONT_SERIF
    elif italic:
        path = FONT_SANS_ITALIC
    else:
        path = FONT_SANS_BOLD if bold else FONT_SANS
    return ImageFont.truetype(str(path), size=size)


def read_csv(relative: str) -> list[dict[str, str]]:
    normalized = relative.replace("\\", "/")
    lower = f"/{normalized.lower()}/"
    assert normalized in SAFE_INPUTS, f"Input is not allow-listed: {relative}"
    assert not any(part in lower for part in FORBIDDEN_PARTS), f"Unsafe path refused: {relative}"
    with (ROOT / normalized).open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def as_float(value: str | float | None) -> float | None:
    try:
        result = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def as_bool(value: str | bool | None) -> bool:
    return value is True or str(value).strip().lower() == "true"


def enrich(row: dict[str, object]) -> dict[str, object] | None:
    effect_type = str(row.get("effect_type", ""))
    effect = as_float(row.get("effect"))
    ci_low = as_float(row.get("ci_low"))
    ci_high = as_float(row.get("ci_high"))
    p = as_float(row.get("p"))
    if effect_type not in {"OR", "IRR", "beta"} or effect is None or ci_low is None or ci_high is None:
        return None
    if effect_type in {"OR", "IRR"}:
        if min(effect, ci_low, ci_high) <= 0:
            return None
        signed = math.log2(effect)
        signed_low = math.log2(ci_low)
        signed_high = math.log2(ci_high)
    else:
        signed = effect
        signed_low = ci_low
        signed_high = ci_high
    plot_y = Y_CAP if p is None or p <= 0 else min(-math.log10(max(p, 1e-300)), Y_CAP)
    result = dict(row)
    result.update(
        effect=effect,
        ci_low=ci_low,
        ci_high=ci_high,
        p=p,
        signed=signed,
        signed_low=signed_low,
        signed_high=signed_high,
        magnitude=abs(signed),
        plot_y=plot_y,
        sig_fdr=as_bool(row.get("sig_fdr")),
        sig_bon=as_bool(row.get("sig_bon")),
        n_value=as_float(row.get("n")),
    )
    return result


def primary_rows(relative: str) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for source in read_csv(relative):
        if source.get("contrast") != "ahi_ge5" or source.get("model") != "m4":
            continue
        item = enrich(dict(source))
        if item is not None:
            output.append(item)
    return output


def smart_title(value: str) -> str:
    value = value.strip()
    if value and value == value.upper():
        value = value.title()
    replacements = {
        "Cns": "CNS",
        "Ii": "II",
        "Iii": "III",
        "Iv": "IV",
        "Nos": "NOS",
        "Ed": "ED",
        "Ecg": "ECG",
    }
    for old, new in replacements.items():
        value = re.sub(rf"\b{re.escape(old)}\b", new, value)
    return value


def short_proc_label(value: str) -> str:
    exact = {
        "Extern Electrocardiogram Record for > 48 Hrs Up to 21 Days, Cont Record, Storage, Analysis with Rpt, Review & Int": "Ambulatory ECG, >48 h to 21 d",
        "Ext Electrocardiogram >7 Days <= 15 Days, Contin Rhythm Recording Storage, Analysis with Rpt, Review & Int": "Continuous ECG, 7–15 d",
        "Ext Electrocardiogram > 48 Hrs <= 7 Days, Contin Rhythm Recording Storage, Analysis with Rpt, Review & Int": "Continuous ECG, >48 h to 7 d",
        "Electrocardiogram External Pat Activated Event Recording Up to 30 Days, 24 Hour Monitoring, Recording": "Patient-activated ECG, recording",
        "Electrocardiogram External Pat Activated Event Recording Up to 30 Days, 24 Hour Monitoring, Transmit, Review & Int": "Patient-activated ECG, transmission/review",
        "Electrocardiogram External Pat Activated Event Recording Up to 30 Days, 24 Hour Monitoring, Physician Rev & Int": "Patient-activated ECG, physician review",
        "External Electrocardiogram Up to 48 Hrs Monitor with Recording, Analysis, Rpt, Physician Review & Int": "External ECG monitoring, ≤48 h",
        "External Electrocardiogram 24 Hour Monitor Recording Physician Review & Int": "24-hour ECG monitor, physician review",
        "Electrocardiogram 12 or More Leads Int & Rpt Only": "12-lead ECG interpretation/report",
        "Cv Stress Test Treadmill with Physician Supervision Only": "Treadmill stress test",
        "Cv Stress Test Treadmill with Sup, Int & Rpt": "Treadmill stress test, complete service",
        "Spirometry, Pre and Post Bronchodilator": "Pre/post-bronchodilator spirometry",
        "Apply Splint": "Splint application",
        "Apply Cast": "Cast application",
        "Operative Procedure Scanned Report": "Scanned operative report",
        "Allergy Test, Percut with Allergenic Extracts, Immediate Type": "Percutaneous immediate-allergy testing",
        "Large Joint Injection or Aspiration Procedures": "Large-joint injection or aspiration",
        "Electroencephalogram, Eeg": "Electroencephalography (EEG)",
        "Eeg, Awake and Asleep": "EEG, awake and asleep",
        "Cystoscopy Procedures": "Cystoscopy",
        "Laceration Repair Procedures": "Laceration repair",
        "Airway Management, Anesth Proc": "Anesthesia airway management",
        "Spinal, Anesth Proc": "Spinal anesthesia",
        "Esophagogastroduodenoscopy": "Upper endoscopy (EGD)",
        "Ultrasound Scan of Bladder": "Bladder ultrasound",
        "Measurement of Post Void Residual Urine Volume": "Postvoid residual urine measurement",
        "Measurement, Thoracic Gas Volume": "Thoracic gas-volume measurement",
        "Measurement of Diffusion Lung Capacity": "Lung diffusing-capacity measurement",
    }
    if value in exact:
        return exact[value]
    value = value.replace(" with Rpt, Review & Int", "")
    value = value.replace("Review & Int", "review")
    value = value.replace("Physician", "")
    value = re.sub(r"\s+", " ", value).strip(" ,")
    return value


def q_label(feature: str, question_map: dict[str, str]) -> str:
    manual = {
        "ESS_TOTAL": "Epworth Sleepiness Scale total",
        "ISI_TOTAL": "Insomnia Severity Index total",
        "PHQ2_TOTAL": "PHQ-2 total",
        "FOSQ_MEAN": "FOSQ-10 mean score",
        "GASP": "Awakening short of breath or gasping",
        "MEDS": "Sleep-medication use",
        "ISI5": "Sleep problem noticeable to others",
        "CC_SNORING_APNEA": "Chief complaint: snoring or apnea",
        "CC_INSOMNIA": "Chief complaint: insomnia",
        "CC_REFERRAL": "Chief complaint: referral",
        "CC_DRY_MOUTH": "Chief complaint: dry mouth",
        "CC_FATIGUE": "Chief complaint: fatigue",
        "CC_HEADACHE": "Chief complaint: headache",
        "CC_NOCTURIA": "Chief complaint: nocturia",
        "CC_OTHER": "Chief complaint: other",
        "CC_PARASOMNIA": "Chief complaint: parasomnia",
        "CC_RESTLESS_LEGS": "Chief complaint: restless legs",
        "CC_SLEEPINESS": "Chief complaint: sleepiness",
        "RLS2": "Restless-legs symptoms that occur mainly at rest and improve with movement",
        "RLS3": "Restless-legs symptoms that are worse in the evening than in the morning",
    }
    if feature in manual:
        return manual[feature]
    time_match = re.fullmatch(r"(BEDTIME|WAKETIME)_(\d{4})_(\d{4})", feature)
    if time_match:
        kind, start, end = time_match.groups()
        start = f"{start[:2]}:{start[2:]}"
        end = f"{end[:2]}:{end[2:]}"
        return f"{'Bedtime' if kind == 'BEDTIME' else 'Wake time'} {start}–{end}"
    if feature in question_map:
        label = question_map[feature].rstrip("?")
        label = re.sub(r"\s+", " ", label)
        return label
    return smart_title(feature.replace("_", " "))


def load_data() -> dict[str, list[dict[str, object]]]:
    med_map = {
        str(row["code"]).zfill(4): row for row in read_csv("results/medwas_results/gpi_category.csv")
    }
    med: list[dict[str, object]] = []
    for window in ("1yr", "5yr"):
        for row in primary_rows(f"results/medwas_results/analysis/results_{window}.csv"):
            code = str(row["code"]).zfill(4)
            mapping = med_map[code]
            label = smart_title(str(mapping["drug_class_name"])).rstrip(" (")
            label = {
                "Modified Cyclics": "Modified cyclic antidepressants",
                "Calcium Blockers": "Calcium-channel blockers",
                "Angiotensin Ii Receptor Antagonist": "Angiotensin II receptor antagonists",
            }.get(label, label)
            row.update(
                window=window,
                code=code,
                label=label,
                group=smart_title(str(mapping["category"])),
                detail=smart_title(str(mapping["drug_group_name"])),
            )
            med.append(row)

    proc: list[dict[str, object]] = []
    for window in ("1yr", "5yr"):
        for row in primary_rows(f"results/procwas_results/analysis/results_{window}.csv"):
            row.update(
                window=window,
                label=short_proc_label(str(row["feature"])),
                group=smart_title(str(row["body_system"])),
                detail=smart_title(str(row["procedure_type"])),
            )
            proc.append(row)

    util: list[dict[str, object]] = []
    for window in ("1yr", "5yr"):
        for row in primary_rows(f"results/utilwas_results/analysis/results_phecode_{window}.csv"):
            if row.get("combo") not in {"ED_primary", "ED_secondary"}:
                continue
            role = "ED primary diagnosis" if row["combo"] == "ED_primary" else "ED secondary diagnosis"
            estimand = "Presence" if row["part"] == "presence" else "Conditional encounter rate"
            category = smart_title(str(row["category"]))
            row.update(
                window=window,
                label=str(row["phenotype"]),
                role=role,
                estimand=estimand,
                group=f"{'Primary' if row['combo'] == 'ED_primary' else 'Secondary'} · {category}",
                detail=category,
            )
            util.append(row)

    candidates: dict[str, list[str]] = defaultdict(list)
    for row in read_csv("results/questionnaire_results/crosswalk.csv"):
        code = str(row.get("target_code", ""))
        question = re.sub(r"\s+", " ", str(row.get("question_text", "")).strip())
        if code and question and question not in candidates[code]:
            candidates[code].append(question)
    question_map = {
        code: sorted(values, key=lambda value: (len(value), value.lower()))[0]
        for code, values in candidates.items()
    }
    qwas: list[dict[str, object]] = []
    for row in primary_rows("results/qwas_results/analysis/results.csv"):
        feature = str(row["feature"])
        label = q_label(feature, question_map)
        referral = as_bool(row.get("referral_item"))
        row.update(
            window="pre-index",
            label=label + ("*" if referral else ""),
            group=str(row["domain"]),
            detail=f"{row['domain']} · {row['form']}",
            referral=referral,
        )
        qwas.append(row)

    behavior_labels = {
        "adherence_pdc": "Medication PDC among users",
        "adherent": "Medication adherence (PDC ≥0.80)",
        "flu_received": "Influenza vaccination received",
        "flu_count": "Influenza vaccination count",
        "fmla_filed": "Medical-leave claim filed",
        "fmla_dur_days": "Medical-leave duration",
        "fmla_admitted": "Medical-leave claim admitted",
        "fmla_pregnancy": "Pregnancy-related leave",
        "any_noshow": "Any missed appointment",
        "no_show_pct": "Missed-appointment proportion",
        "any_off_work": "Any off-work order",
        "off_work_days": "Total off-work days",
    }
    beh: list[dict[str, object]] = []
    for row in primary_rows("results/behwas_results/analysis/results_all.csv"):
        window = str(row["window"])
        row.update(
            label=behavior_labels[str(row["outcome"])],
            group={"1yr": "1 year", "5yr": "5 years", "lifetime": "Undated / lifetime"}[window],
            detail=str(row["kind"]),
        )
        beh.append(row)

    return {"med": med, "proc": proc, "util": util, "qwas": qwas, "beh": beh}


def percentile(values: Iterable[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    position = (len(ordered) - 1) * q
    low = int(math.floor(position))
    high = int(math.ceil(position))
    if low == high:
        return ordered[low]
    fraction = position - low
    return ordered[low] * (1 - fraction) + ordered[high] * fraction


def marker_radius(row: dict[str, object], rows: list[dict[str, object]], small: int = 5, large: int = 12) -> int:
    values = [float(item["magnitude"]) for item in rows]
    lo = percentile(values, 0.10)
    hi = percentile(values, 0.90)
    value = min(max(float(row["magnitude"]), lo), hi)
    fraction = 0.5 if hi <= lo else (value - lo) / (hi - lo)
    return round(small + math.sqrt(fraction) * (large - small))


def triangle(
    draw: ImageDraw.ImageDraw,
    x: float,
    y: float,
    radius: int,
    direction: int,
    fill: str,
    outline: str | None = None,
    width: int = 1,
) -> None:
    if direction >= 0:
        points = [(x, y - radius), (x - radius, y + radius * 0.8), (x + radius, y + radius * 0.8)]
    else:
        points = [(x - radius, y - radius * 0.8), (x + radius, y - radius * 0.8), (x, y + radius)]
    draw.polygon(points, fill=fill)
    if outline:
        draw.line(points + [points[0]], fill=outline, width=width, joint="curve")


def marker_style(row: dict[str, object]) -> tuple[int, str, str | None, int]:
    direction = 1 if float(row["signed"]) >= 0 else -1
    if as_bool(row["sig_fdr"]):
        fill = TEAL if direction > 0 else PURPLE
        return direction, fill, OUTLINE, 2
    fill = TEAL_LIGHT if direction > 0 else PURPLE_LIGHT
    return direction, fill, "#879594", 1


def draw_header(
    image: Image.Image,
    kicker: str,
    title: str,
    subtitle: str,
    *,
    legend: bool = True,
) -> int:
    draw = ImageDraw.Draw(image)
    draw.text((70, 52), kicker.upper(), fill=TEAL, font=font(21, bold=True))
    draw.text((70, 86), title, fill=INK, font=font(48, serif=True))
    draw.text((72, 151), subtitle, fill=MID_INK, font=font(23))
    draw.line((70, 194, WIDTH - 70, 194), fill="#B8CFCA", width=2)
    if not legend:
        return 225
    y = 239
    triangle(draw, 88, y, 11, 1, TEAL, OUTLINE, 2)
    draw.text((108, y - 14), "positive / higher in AHI ≥5", fill=INK, font=font(20))
    triangle(draw, 425, y, 11, -1, PURPLE, OUTLINE, 2)
    draw.text((445, y - 14), "inverse / lower in AHI ≥5", fill=INK, font=font(20))
    triangle(draw, 805, y, 9, 1, TEAL, OUTLINE, 2)
    draw.text((824, y - 14), "FDR-significant (dark outline)", fill=INK, font=font(20))
    triangle(draw, 1197, y, 9, 1, TEAL_LIGHT, "#879594", 1)
    draw.text((1216, y - 14), "FDR-null (muted)", fill=INK, font=font(20))
    triangle(draw, 1540, y, 6, 1, MUTED)
    triangle(draw, 1590, y, 10, 1, MUTED)
    triangle(draw, 1650, y, 14, 1, MUTED)
    draw.text((1680, y - 14), "larger = greater |effect|", fill=INK, font=font(19))
    return 285


def wrap_to_width(draw: ImageDraw.ImageDraw, text: str, typeface: ImageFont.FreeTypeFont, width: int) -> list[str]:
    words = text.split()
    if not words:
        return [""]
    lines: list[str] = []
    current = words[0]
    for word in words[1:]:
        candidate = current + " " + word
        if draw.textbbox((0, 0), candidate, font=typeface)[2] <= width:
            current = candidate
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines


def shortened(text: str, limit: int = 64) -> str:
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int = 5) -> bool:
    return not (a[2] + pad < b[0] or b[2] + pad < a[0] or a[3] + pad < b[1] or b[3] + pad < a[1])


def annotate_point(
    draw: ImageDraw.ImageDraw,
    point: tuple[int, int],
    label: str,
    plot_box: tuple[int, int, int, int],
    occupied: list[tuple[int, int, int, int]],
    index: int,
) -> None:
    label_font = font(18, bold=True)
    max_width = 205
    lines = wrap_to_width(draw, shortened(label, 58), label_font, max_width)
    lines = lines[:3]
    line_height = 21
    text_width = max(draw.textbbox((0, 0), line, font=label_font)[2] for line in lines)
    box_w = text_width + 14
    box_h = len(lines) * line_height + 10
    px, py = point
    x0, y0, x1, y1 = plot_box
    candidates = []
    for vertical_offset in (-box_h - 14, 14, 58, 102, -box_h - 58):
        candidates.extend(
            [
                (px + 16, py + vertical_offset),
                (px - box_w - 16, py + vertical_offset),
                (px - box_w / 2, py + vertical_offset),
            ]
        )
    candidates = candidates[index % len(candidates) :] + candidates[: index % len(candidates)]
    chosen: tuple[int, int, int, int] | None = None
    for bx, by in candidates:
        bx = min(max(bx, x0 + 2), x1 - box_w - 2)
        by = min(max(by, y0 + 2), y1 - box_h - 2)
        box = (int(bx), int(by), int(bx + box_w), int(by + box_h))
        if not any(overlap(box, old) for old in occupied):
            chosen = box
            break
    if chosen is None:
        bx, by = candidates[0]
        bx = min(max(bx, x0 + 2), x1 - box_w - 2)
        by = min(max(by, y0 + 2), y1 - box_h - 2)
        chosen = (int(bx), int(by), int(bx + box_w), int(by + box_h))
    occupied.append(chosen)
    bx0, by0, bx1, by1 = chosen
    anchor_x = bx0 if px < (bx0 + bx1) / 2 else bx1
    anchor_y = min(max(py, by0 + 4), by1 - 4)
    draw.line((px, py, anchor_x, anchor_y), fill="#627978", width=2)
    draw.rounded_rectangle(chosen, radius=5, fill=WHITE, outline="#9CB0AD", width=1)
    for line_no, line in enumerate(lines):
        draw.text((bx0 + 7, by0 + 4 + line_no * line_height), line, fill=INK, font=label_font)


def choose_key_labels(rows: list[dict[str, object]], max_labels: int = 3) -> list[dict[str, object]]:
    selected: list[dict[str, object]] = []
    for direction in (1, -1):
        candidates = [
            row
            for row in rows
            if as_bool(row["sig_fdr"]) and (1 if float(row["signed"]) >= 0 else -1) == direction
        ]
        candidates.sort(key=lambda row: (float("inf") if row["p"] is None else float(row["p"]), -float(row["magnitude"])))
        selected.extend(candidates[:2])
    if len(selected) < max_labels:
        remaining = sorted(
            [row for row in rows if row not in selected],
            key=lambda row: (
                not as_bool(row["sig_fdr"]),
                float("inf") if row["p"] is None else float(row["p"]),
                -float(row["magnitude"]),
            ),
        )
        selected.extend(remaining[: max_labels - len(selected)])
    return selected[:max_labels]


def group_label(value: str) -> str:
    replacements = {
        "Analgesics & Anesthetics": "Analgesic/anesth.",
        "Anti-Infective Agents": "Anti-infective",
        "Antineoplastic Agents": "Antineoplastic",
        "Biologicals": "Biologics",
        "Endocrine & Metabolic": "Endocrine/metab.",
        "Gastrointestinal": "GI",
        "Genitourinary": "GU",
        "Hematological": "Hematologic",
        "General / Not System-Specific": "General",
        "General / Not system-specific": "General",
        "Cardiovascular": "Cardio",
        "Cardiovascular / Lymphatic": "Cardio",
        "Stimulants/Anti-Obesity": "Stimulant/obesity",
        "Misc Psych/Neuro": "Psych / neuro",
        "Miscellaneous": "Misc.",
        "Nutritional": "Nutrition",
        "Neuromuscular": "Neuromuscular",
        "Respiratory": "Respiratory",
        "Eye / Ophthalmic": "Ophthalmic",
        "Female reproductive": "Reproductive",
        "Integumentary / Skin": "Skin",
        "Maternity / Obstetric": "Obstetric",
        "Nervous system": "Nervous",
        "Urinary / Renal": "Urinary / renal",
        "circulatory system": "Circulatory",
        "congenital anomalies": "Congenital",
        "dermatologic": "Dermatologic",
        "endocrine/metabolic": "Endocrine/metab.",
        "hematopoietic": "Hematologic",
        "infectious diseases": "Infectious",
        "injuries & poisonings": "Injury/poison",
        "mental disorders": "Mental health",
        "musculoskeletal": "Musculoskeletal",
        "neurological": "Neurologic",
        "pregnancy complications": "Pregnancy",
        "sense organs": "Sense organs",
    }
    value = replacements.get(value, value)
    return shortened(value, 20)


def forest_group_label(value: object) -> str:
    text = str(value)
    replacements = {
        "cardiovascular / lymphatic": "cardiovascular",
        "general / not system-specific": "general",
        "endocrine/metabolic": "endocrine/metabolic",
        "injuries & poisonings": "injury/poisoning",
        "mental disorders": "mental health",
        "circulatory system": "circulatory",
        "sense organs": "sense organs",
    }
    return replacements.get(text.lower(), text)


def draw_rotated_centered(
    image: Image.Image,
    text: str,
    center: tuple[int, int],
    typeface: ImageFont.FreeTypeFont,
    fill: str,
    angle: float = 58,
) -> None:
    box = typeface.getbbox(text)
    temp = Image.new("RGBA", (max(1, box[2] - box[0] + 10), max(1, box[3] - box[1] + 10)), (0, 0, 0, 0))
    temp_draw = ImageDraw.Draw(temp)
    temp_draw.text((5 - box[0], 5 - box[1]), text, font=typeface, fill=fill)
    rotated = temp.rotate(angle, expand=True, resample=Image.Resampling.BICUBIC)
    image.paste(rotated, (int(center[0] - rotated.width / 2), int(center[1] - rotated.height / 2)), rotated)


def ordered_rows(rows: list[dict[str, object]]) -> tuple[list[dict[str, object]], list[tuple[str, int, int]]]:
    groups = sorted({str(row["group"]) for row in rows}, key=lambda value: value.lower())
    ordered: list[dict[str, object]] = []
    spans: list[tuple[str, int, int]] = []
    for group in groups:
        start = len(ordered)
        group_rows = sorted((row for row in rows if str(row["group"]) == group), key=lambda row: str(row["label"]).lower())
        ordered.extend(group_rows)
        spans.append((group, start, len(ordered)))
    return ordered, spans


def render_manhattan_panel(
    image: Image.Image,
    box: tuple[int, int, int, int],
    title: str,
    rows: list[dict[str, object]],
) -> None:
    draw = ImageDraw.Draw(image)
    x0, y0, x1, y1 = box
    draw.rounded_rectangle(box, radius=12, fill=PANEL_BG, outline="#C7D5D2", width=2)
    draw.text((x0 + 22, y0 + 18), title, fill=INK, font=font(27, bold=True))
    summary = f"{len(rows):,} estimates · {sum(as_bool(row['sig_fdr']) for row in rows):,} FDR-significant"
    draw.text((x0 + 24, y0 + 54), summary, fill=MID_INK, font=font(19))
    plot = (x0 + 92, y0 + 94, x1 - 22, y1 - 150)
    px0, py0, px1, py1 = plot
    ordered, spans = ordered_rows(rows)
    count = max(len(ordered), 1)
    observed_y = max((float(row["plot_y"]) for row in rows), default=0.0)
    y_max = Y_CAP
    if observed_y < 45:
        y_max = next(
            candidate
            for candidate in (2.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 40.0, 50.0)
            if candidate >= max(2.0, observed_y * 1.12)
        )
    for span_index, (group, start, end) in enumerate(spans):
        left = px0 + (px1 - px0) * start / count
        right = px0 + (px1 - px0) * end / count
        draw.rectangle((left, py0, right, py1), fill=BAND_A if span_index % 2 == 0 else BAND_B)
        draw.line((right, py0, right, py1), fill="#D3DDDB", width=1)
        band_center = (left + right) / 2
        label_center = px0 + (px1 - px0) * (span_index + 0.5) / len(spans)
        draw.line((band_center, py1 + 2, label_center, py1 + 24), fill="#A8B7B5", width=1)
        draw_rotated_centered(
            image,
            group_label(group),
            (int(label_center), py1 + 72),
            font(19, bold=True),
            MID_INK,
        )
    for tick_index in range(6):
        tick = y_max * tick_index / 5
        y = py1 - (py1 - py0) * tick / y_max
        draw.line((px0, y, px1, y), fill=GRID, width=1)
        label = f"{tick:g}"
        label_box = draw.textbbox((0, 0), label, font=font(17))
        draw.text((px0 - 12 - (label_box[2] - label_box[0]), y - 10), label, fill=MID_INK, font=font(17))
    draw.line((px0, py0, px0, py1), fill=INK, width=2)
    draw.line((px0, py1, px1, py1), fill=INK, width=2)
    coordinates: dict[int, tuple[int, int]] = {}
    for index, row in enumerate(ordered):
        x = px0 + (px1 - px0) * (index + 0.5) / count
        y = py1 - (py1 - py0) * float(row["plot_y"]) / y_max
        radius = marker_radius(row, rows)
        direction, fill, outline, line_width = marker_style(row)
        triangle(draw, x, y, radius, direction, fill, outline, line_width)
        coordinates[id(row)] = (round(x), round(y))
    occupied: list[tuple[int, int, int, int]] = []
    for index, row in enumerate(choose_key_labels(rows)):
        annotate_point(draw, coordinates[id(row)], str(row["label"]), plot, occupied, index)
    draw_rotated_centered(
        image,
        "-log10(P)",
        (x0 + 25, int((py0 + py1) / 2)),
        font(18, bold=True),
        MID_INK,
        angle=90,
    )


def render_volcano_panel(
    image: Image.Image,
    box: tuple[int, int, int, int],
    title: str,
    rows: list[dict[str, object]],
    xlabel: str,
) -> None:
    draw = ImageDraw.Draw(image)
    x0, y0, x1, y1 = box
    draw.rounded_rectangle(box, radius=12, fill=PANEL_BG, outline="#C7D5D2", width=2)
    draw.text((x0 + 22, y0 + 18), title, fill=INK, font=font(27, bold=True))
    summary = f"{len(rows):,} estimates · {sum(as_bool(row['sig_fdr']) for row in rows):,} FDR-significant"
    draw.text((x0 + 24, y0 + 54), summary, fill=MID_INK, font=font(19))
    plot = (x0 + 94, y0 + 92, x1 - 26, y1 - 83)
    px0, py0, px1, py1 = plot
    max_abs = max((abs(float(row["signed"])) for row in rows), default=0.1)
    bound = max(max_abs * 1.12, 0.10)
    observed_y = max((float(row["plot_y"]) for row in rows), default=0.0)
    y_max = Y_CAP
    if observed_y < 45:
        y_max = next(
            candidate
            for candidate in (2.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 40.0, 50.0)
            if candidate >= max(2.0, observed_y * 1.12)
        )
    for tick_index in range(6):
        tick = y_max * tick_index / 5
        y = py1 - (py1 - py0) * tick / y_max
        draw.line((px0, y, px1, y), fill=GRID, width=1)
        label = f"{tick:g}"
        label_box = draw.textbbox((0, 0), label, font=font(17))
        draw.text((px0 - 12 - (label_box[2] - label_box[0]), y - 10), label, fill=MID_INK, font=font(17))
    for fraction in (-1, -0.5, 0, 0.5, 1):
        x = px0 + (fraction + 1) * (px1 - px0) / 2
        draw.line((x, py0, x, py1), fill="#819391" if fraction == 0 else GRID, width=2 if fraction == 0 else 1)
        value = fraction * bound
        text = f"{value:.2f}" if bound < 0.5 else f"{value:.1f}"
        tbox = draw.textbbox((0, 0), text, font=font(16))
        draw.text((x - (tbox[2] - tbox[0]) / 2, py1 + 8), text, fill=MID_INK, font=font(16))
    draw.line((px0, py0, px0, py1), fill=INK, width=2)
    draw.line((px0, py1, px1, py1), fill=INK, width=2)
    coordinates: dict[int, tuple[int, int]] = {}
    for row in rows:
        x = px0 + (float(row["signed"]) + bound) / (2 * bound) * (px1 - px0)
        y = py1 - (py1 - py0) * float(row["plot_y"]) / y_max
        radius = marker_radius(row, rows)
        direction, fill, outline, line_width = marker_style(row)
        triangle(draw, x, y, radius, direction, fill, outline, line_width)
        coordinates[id(row)] = (round(x), round(y))
    occupied: list[tuple[int, int, int, int]] = []
    for index, row in enumerate(choose_key_labels(rows)):
        annotate_point(draw, coordinates[id(row)], str(row["label"]), plot, occupied, index)
    label_box = draw.textbbox((0, 0), xlabel, font=font(18, bold=True))
    draw.text(((px0 + px1 - (label_box[2] - label_box[0])) / 2, y1 - 31), xlabel, fill=MID_INK, font=font(18, bold=True))
    draw_rotated_centered(
        image,
        "-log10(P)",
        (x0 + 25, int((py0 + py1) / 2)),
        font(18, bold=True),
        MID_INK,
        angle=90,
    )


def save_page(image: Image.Image, filename: str) -> Path:
    path = OUT / filename
    image.save(path, format="PNG", dpi=DPI, optimize=True)
    return path


def grid_boxes(top: int, bottom: int, columns: int, rows: int, gap: int = 28) -> list[tuple[int, int, int, int]]:
    left = 60
    right = WIDTH - 60
    cell_w = (right - left - gap * (columns - 1)) / columns
    cell_h = (bottom - top - gap * (rows - 1)) / rows
    boxes = []
    for row in range(rows):
        for column in range(columns):
            x0 = round(left + column * (cell_w + gap))
            y0 = round(top + row * (cell_h + gap))
            boxes.append((x0, y0, round(x0 + cell_w), round(y0 + cell_h)))
    return boxes


def build_manhattan_and_volcano(data: dict[str, list[dict[str, object]]]) -> list[dict[str, object]]:
    manifest: list[dict[str, object]] = []
    med = data["med"]
    proc = data["proc"]
    util = data["util"]
    qwas = data["qwas"]

    def add_chart_page(
        *,
        figure: str,
        filename: str,
        kicker_domain: str,
        title: str,
        subtitle: str,
        panels: list[tuple[str, list[dict[str, object]], str]],
        chart: str,
        domain: str,
        effect_type: str | list[str],
        selection_rule: str,
        caption_notes: str,
        footer: str = "",
    ) -> None:
        assert 1 <= len(panels) <= 2
        image = Image.new("RGB", (WIDTH, 2500), BG)
        top = draw_header(image, f"Figure {figure} · {kicker_domain}", title, subtitle)
        boxes = grid_boxes(top, 2405 if footer else 2440, 1, len(panels))
        for box, (panel_title, rows, xlabel) in zip(boxes, panels):
            if chart == "manhattan":
                render_manhattan_panel(image, box, panel_title, rows)
            else:
                render_volcano_panel(image, box, panel_title, rows, xlabel)
        if footer:
            ImageDraw.Draw(image).text((72, 2435), footer, fill=MID_INK, font=font(21, italic=True))
        path = save_page(image, filename)
        manifest.append(
            {
                "figure": figure,
                "file": path.name,
                "width": WIDTH,
                "height": 2500,
                "rows": sum(len(panel[1]) for panel in panels),
                "domain": domain,
                "effect_type": effect_type,
                "selection_rule": selection_rule,
                "caption_notes": caption_notes,
                "panel_rows": {panel[0]: len(panel[1]) for panel in panels},
            }
        )

    med_panels = [
        ("1-year medication fills", [row for row in med if row["window"] == "1yr"], "log2(OR)"),
        ("5-year medication fills", [row for row in med if row["window"] == "5yr"], "log2(OR)"),
    ]
    proc_panels = [
        ("1-year procedure rates", [row for row in proc if row["window"] == "1yr"], "log2(IRR)"),
        ("5-year procedure rates", [row for row in proc if row["window"] == "5yr"], "log2(IRR)"),
    ]
    add_chart_page(
        figure="5A",
        filename="figure5a_medwas_manhattan.png",
        kicker_domain="Medication-wide association study",
        title="Medication associations across therapeutic subdomains",
        subtitle="Fully adjusted m4 estimates compare AHI ≥5 with AHI <5; shaded bands identify medication categories.",
        panels=med_panels,
        chart="manhattan",
        domain="MedWAS",
        effect_type="OR",
        selection_rule="All 177 one-year and 218 five-year primary MedWAS rows",
        caption_notes="Y is -log10(P), globally capped at 50; alternating bands are medication categories; point size is within-panel absolute log2(OR); labels identify leading positive and inverse associations.",
    )
    add_chart_page(
        figure="5B",
        filename="figure5b_procwas_manhattan.png",
        kicker_domain="Procedure-wide association study",
        title="Procedure-rate associations across body systems",
        subtitle="Fully adjusted m4 estimates compare AHI ≥5 with AHI <5; shaded bands identify procedure body systems.",
        panels=proc_panels,
        chart="manhattan",
        domain="ProcWAS",
        effect_type="IRR",
        selection_rule="All 25 one-year and 67 five-year primary ProcWAS rows",
        caption_notes="Y is -log10(P), globally capped at 50; alternating bands are procedure body systems; point size is within-panel absolute log2(IRR); labels identify leading associations.",
    )

    def util_window_panels(window: str, part: str) -> list[tuple[str, list[dict[str, object]], str]]:
        panels = []
        for combo in ("ED_primary", "ED_secondary"):
            rows = [
                {**row, "group": row["detail"]}
                for row in util
                if row["window"] == window and row["part"] == part and row["combo"] == combo
            ]
            role = "Primary ED diagnoses" if combo == "ED_primary" else "Secondary ED diagnoses"
            panels.append((role, rows, "log2(OR)" if part == "presence" else "log2(IRR)"))
        return panels

    for figure, window, part in [
        ("6A1", "1yr", "presence"),
        ("6A2", "5yr", "presence"),
        ("6B1", "1yr", "count_among_present"),
        ("6B2", "5yr", "count_among_present"),
    ]:
        presence = part == "presence"
        window_text = "1-year" if window == "1yr" else "5-year"
        add_chart_page(
            figure=figure,
            filename=f"figure{figure.lower()}_utilwas_ed_{'presence' if presence else 'conditional'}_manhattan.png",
            kicker_domain="Emergency-department utilization-wide association study",
            title=(f"{window_text} presence of ED diagnoses" if presence else f"{window_text} ED encounter rates among patients with each diagnosis"),
            subtitle=(
                "Odds ratios model any qualifying ED diagnosis in the full cohort; primary and secondary roles are separate."
                if presence
                else "Incidence-rate ratios are conditional on ≥1 qualifying ED encounter for the diagnosis; primary and secondary roles are separate."
            ),
            panels=util_window_panels(window, part),
            chart="manhattan",
            domain="UtilWAS, ED visits only",
            effect_type="OR" if presence else "IRR",
            selection_rule=f"All {window_text} ED-primary and ED-secondary {'diagnosis-presence' if presence else 'conditional-rate'} rows",
            caption_notes=(
                "Y is -log10(P), globally capped at 50; shaded bands are Phecode categories; point size is within-panel absolute log2(OR); presence is a full-cohort estimand and must not be interpreted as a conditional rate."
                if presence
                else "Y is -log10(P), globally capped at 50; shaded bands are Phecode categories; point size is within-panel absolute log2(IRR); analytic n is diagnosis-specific and conditional on ≥1 qualifying ED encounter."
            ),
        )

    qwas_panels = [
        ("Binary questionnaire items", [row for row in qwas if row["form"] == "binary"], "log2(OR)"),
        ("Continuous questionnaire items", [row for row in qwas if row["form"] == "continuous"], "β (rank-inverse-normal SD)"),
    ]
    q_footer = "* Referral-driving questionnaire item; interpretation is ascertainment-sensitive."
    add_chart_page(
        figure="6C",
        filename="figure6c_qwas_manhattan.png",
        kicker_domain="Questionnaire-wide association study",
        title="Patient-reported associations across symptom domains",
        subtitle="Binary and continuous forms are displayed separately; shaded bands identify questionnaire domains.",
        panels=qwas_panels,
        chart="manhattan",
        domain="QWAS",
        effect_type=["OR", "beta"],
        selection_rule="All 83 binary and 49 continuous primary QWAS rows",
        caption_notes="Y is -log10(P), globally capped at 50; point size is within-panel absolute log2(OR) or absolute beta; asterisks mark referral-driving, ascertainment-sensitive items; beta is in rank-inverse-normal SD units.",
        footer=q_footer,
    )

    add_chart_page(
        figure="7A",
        filename="figure7a_medwas_volcano.png",
        kicker_domain="Medication-wide association study",
        title="Direction, magnitude, and statistical strength of medication associations",
        subtitle="Vertical position is -log10(P), capped at 50; horizontal position is signed log2(OR).",
        panels=med_panels,
        chart="volcano",
        domain="MedWAS",
        effect_type="OR",
        selection_rule="All 177 one-year and 218 five-year primary MedWAS rows",
        caption_notes="Point size is within-panel absolute log2(OR); labels identify leading positive and inverse associations.",
    )
    add_chart_page(
        figure="7B",
        filename="figure7b_procwas_volcano.png",
        kicker_domain="Procedure-wide association study",
        title="Direction, magnitude, and statistical strength of procedure-rate associations",
        subtitle="Vertical position is -log10(P), capped at 50; horizontal position is signed log2(IRR).",
        panels=proc_panels,
        chart="volcano",
        domain="ProcWAS",
        effect_type="IRR",
        selection_rule="All 25 one-year and 67 five-year primary ProcWAS rows",
        caption_notes="Point size is within-panel absolute log2(IRR); labels identify leading associations.",
    )

    for figure, window, part in [
        ("8A1", "1yr", "presence"),
        ("8A2", "5yr", "presence"),
        ("8B1", "1yr", "count_among_present"),
        ("8B2", "5yr", "count_among_present"),
    ]:
        presence = part == "presence"
        window_text = "1-year" if window == "1yr" else "5-year"
        add_chart_page(
            figure=figure,
            filename=f"figure{figure.lower()}_utilwas_ed_{'presence' if presence else 'conditional'}_volcano.png",
            kicker_domain="Emergency-department utilization-wide association study",
            title=(f"Direction and strength of {window_text} ED diagnosis-presence associations" if presence else f"Direction and strength of {window_text} conditional ED encounter rates"),
            subtitle=(
                "Odds ratios model any qualifying ED diagnosis in the full cohort; primary and secondary roles are separate."
                if presence
                else "Incidence-rate ratios are conditional on ≥1 qualifying ED encounter for the diagnosis; primary and secondary roles are separate."
            ),
            panels=util_window_panels(window, part),
            chart="volcano",
            domain="UtilWAS, ED visits only",
            effect_type="OR" if presence else "IRR",
            selection_rule=f"All {window_text} ED-primary and ED-secondary {'diagnosis-presence' if presence else 'conditional-rate'} rows",
            caption_notes=f"Y is -log10(P), capped at 50; x is log2({'OR' if presence else 'IRR'}); point size is within-panel absolute log2 ratio.",
        )

    add_chart_page(
        figure="8C",
        filename="figure8c_qwas_volcano.png",
        kicker_domain="Questionnaire-wide association study",
        title="Direction, magnitude, and statistical strength of patient-reported associations",
        subtitle="Binary odds ratios and continuous beta estimates are separate; -log10(P) is capped at 50.",
        panels=qwas_panels,
        chart="volcano",
        domain="QWAS",
        effect_type=["OR", "beta"],
        selection_rule="All 83 binary and 49 continuous primary QWAS rows",
        caption_notes="Binary x is log2(OR), continuous x is rank-inverse-normal beta; point size is within-panel effect magnitude; asterisks mark referral-driving items.",
        footer=q_footer,
    )
    return manifest


def nice_step(span: float, target_ticks: int = 6) -> float:
    raw = max(span / target_ticks, 1e-9)
    power = 10 ** math.floor(math.log10(raw))
    normalized = raw / power
    if normalized <= 1:
        factor = 1
    elif normalized <= 2:
        factor = 2
    elif normalized <= 5:
        factor = 5
    else:
        factor = 10
    return factor * power


def forest_ticks(rows: list[dict[str, object]], effect_type: str) -> tuple[float, float, list[tuple[float, str]]]:
    lows = [float(row["signed_low"]) for row in rows]
    highs = [float(row["signed_high"]) for row in rows]
    lo = min(min(lows), 0.0)
    hi = max(max(highs), 0.0)
    pad = max((hi - lo) * 0.08, 0.03)
    lo -= pad
    hi += pad
    if effect_type in {"OR", "IRR"}:
        candidates = [0.20, 0.25, 0.33, 0.50, 0.75, 1.00, 1.25, 1.50, 2.00, 3.00, 4.00, 5.00]
        ticks = [(math.log2(value), f"{value:g}") for value in candidates if lo <= math.log2(value) <= hi]
        if not any(label == "1" for _, label in ticks):
            ticks.append((0.0, "1"))
        ticks.sort()
    else:
        step = nice_step(hi - lo)
        start = math.floor(lo / step) * step
        end = math.ceil(hi / step) * step
        values = []
        current = start
        while current <= end + step / 2:
            if lo <= current <= hi:
                values.append(current)
            current += step
        decimals = max(0, min(3, -math.floor(math.log10(step))))
        ticks = [(value, f"{value:.{decimals}f}") for value in values]
    return lo, hi, ticks


def draw_forest_page(
    filename: str,
    kicker: str,
    title: str,
    subtitle: str,
    rows: list[dict[str, object]],
    effect_type: str,
    context: Callable[[dict[str, object]], str],
    band_key: Callable[[dict[str, object]], str],
    footer: str = "",
) -> dict[str, object]:
    row_count = len(rows)
    label_left = 72
    context_x = 650
    plot_left = 950
    plot_right = 1515
    stat_x = 1545
    label_font_size = 30
    label_font = font(label_font_size, bold=True)
    context_font = font(28)
    stat_font = font(28)

    def context_lines(draw: ImageDraw.ImageDraw, row: dict[str, object]) -> list[str]:
        text = context(row)
        parts = [part.strip() for part in text.split("|") if part.strip()]
        output: list[str] = []
        for part in parts:
            output.extend(wrap_to_width(draw, part, context_font, plot_left - context_x - 24))
        return output or [""]

    probe = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    maximum_lines = max(
        (
            max(
                len(wrap_to_width(probe, str(row["label"]), label_font, context_x - label_left - 32)),
                len(context_lines(probe, row)),
            )
            for row in rows
        ),
        default=1,
    )
    row_pixels = 68 + max(0, maximum_lines - 2) * 30
    height = max(1500, min(2600, 650 + row_count * row_pixels))
    image = Image.new("RGB", (WIDTH, height), BG)
    draw = ImageDraw.Draw(image)
    top = draw_header(image, kicker, title, subtitle, legend=False)
    header_y = top + 12
    draw.text((label_left, header_y), "ASSOCIATION", fill=TEAL, font=font(22, bold=True))
    draw.text((context_x, header_y), "CONTEXT", fill=TEAL, font=font(22, bold=True))
    draw.text((plot_left, header_y), f"{effect_type} (95% CI)", fill=TEAL, font=font(22, bold=True))
    draw.text((stat_x, header_y), "ESTIMATE (95% CI)", fill=TEAL, font=font(22, bold=True))
    table_top = header_y + 52
    table_bottom = height - 165
    row_height = (table_bottom - table_top) / max(row_count, 1)
    lo, hi, ticks = forest_ticks(rows, effect_type)

    previous_band: str | None = None
    band_index = -1
    for index, row in enumerate(rows):
        key = band_key(row)
        if key != previous_band:
            band_index += 1
            previous_band = key
        y0 = table_top + index * row_height
        y1 = table_top + (index + 1) * row_height
        if band_index % 2 == 0:
            draw.rectangle((60, y0, WIDTH - 60, y1), fill=BAND_A)
        draw.line((60, y1, WIDTH - 60, y1), fill="#E0E7E5", width=1)

    for tick_value, tick_label in ticks:
        x = plot_left + (tick_value - lo) / (hi - lo) * (plot_right - plot_left)
        is_null = abs(tick_value) < 1e-12
        draw.line((x, table_top, x, table_bottom), fill="#637674" if is_null else GRID, width=3 if is_null else 1)
        tbox = draw.textbbox((0, 0), tick_label, font=font(22))
        draw.text((x - (tbox[2] - tbox[0]) / 2, table_bottom + 10), tick_label, fill=MID_INK, font=font(22))

    for index, row in enumerate(rows):
        y = table_top + (index + 0.5) * row_height
        lines = wrap_to_width(draw, str(row["label"]), label_font, context_x - label_left - 32)
        line_h = label_font_size + 3
        start_y = y - len(lines) * line_h / 2
        for line_number, line in enumerate(lines):
            draw.text((label_left, start_y + line_number * line_h), line, fill=INK, font=label_font)
        ctx_lines = context_lines(draw, row)
        for line_number, line in enumerate(ctx_lines):
            draw.text((context_x, y - len(ctx_lines) * 31 / 2 + line_number * 31), line, fill=MID_INK, font=context_font)

        x_low = plot_left + (float(row["signed_low"]) - lo) / (hi - lo) * (plot_right - plot_left)
        x_mid = plot_left + (float(row["signed"]) - lo) / (hi - lo) * (plot_right - plot_left)
        x_high = plot_left + (float(row["signed_high"]) - lo) / (hi - lo) * (plot_right - plot_left)
        significant = as_bool(row["sig_fdr"])
        direction = 1 if float(row["signed"]) >= 0 else -1
        point_fill = (TEAL if direction > 0 else PURPLE) if significant else (TEAL_LIGHT if direction > 0 else PURPLE_LIGHT)
        draw.line((x_low, y, x_high, y), fill="#536967" if significant else "#9CA9A7", width=4 if significant else 3)
        draw.line((x_low, y - 7, x_low, y + 7), fill="#536967", width=2)
        draw.line((x_high, y - 7, x_high, y + 7), fill="#536967", width=2)
        triangle(
            draw,
            x_mid,
            y,
            10,
            direction,
            point_fill,
            OUTLINE if significant else "#7F8E8C",
            2 if significant else 1,
        )
        effect = float(row["effect"])
        ci_low = float(row["ci_low"])
        ci_high = float(row["ci_high"])
        if effect_type in {"OR", "IRR"}:
            estimate = f"{effect:.2f} ({ci_low:.2f}–{ci_high:.2f})"
        else:
            estimate = f"{effect:+.3f} ({ci_low:+.3f}–{ci_high:+.3f})"
        if significant:
            estimate += "  FDR"
        draw.text((stat_x, y - 16), estimate, fill=INK if significant else MID_INK, font=stat_font)

    axis_label = f"{effect_type} (log scale)" if effect_type in {"OR", "IRR"} else "β (rank-inverse-normal SD)"
    axis_box = draw.textbbox((0, 0), axis_label, font=font(24, bold=True))
    draw.text(((plot_left + plot_right - (axis_box[2] - axis_box[0])) / 2, table_bottom + 52), axis_label, fill=INK, font=font(24, bold=True))
    if footer:
        draw.text((72, height - 58), footer, fill=MID_INK, font=font(21, italic=True))
    path = save_page(image, filename)
    return {"file": path.name, "width": WIDTH, "height": height, "rows": row_count, "effect_type": effect_type}


def choose_med_forest(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    by_code: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_code[str(row["code"])].append(row)
    candidates = []
    for code, values in by_code.items():
        if {str(row["window"]) for row in values} != {"1yr", "5yr"}:
            continue
        best = min(values, key=lambda row: float("inf") if row["p"] is None else float(row["p"]))
        if not any(as_bool(row["sig_fdr"]) for row in values):
            continue
        direction = 1 if float(best["signed"]) >= 0 else -1
        candidates.append((code, values, direction, float("inf") if best["p"] is None else float(best["p"])))
    selected = []
    for direction in (1, -1):
        selected.extend(sorted((item for item in candidates if item[2] == direction), key=lambda item: item[3])[:10])
    selected.sort(key=lambda item: (0 if item[2] > 0 else 1, item[3]))
    output = []
    for _, values, _, _ in selected:
        output.extend(sorted(values, key=lambda row: 0 if row["window"] == "1yr" else 1))
    assert len(output) == 40, f"Expected 40 paired MedWAS rows; got {len(output)}"
    return output


def build_forests(data: dict[str, list[dict[str, object]]]) -> list[dict[str, object]]:
    manifest: list[dict[str, object]] = []
    med = data["med"]
    proc = data["proc"]
    util = data["util"]
    qwas = data["qwas"]
    beh = data["beh"]

    def add_pages(
        *,
        labels: list[str],
        filenames: list[str],
        page_rows: list[list[dict[str, object]]],
        kicker_domain: str,
        title: str,
        subtitle: str,
        effect_type: str,
        context: Callable[[dict[str, object]], str],
        band_key: Callable[[dict[str, object]], str],
        domain: str,
        selection_rule: str,
        caption_notes: str,
        footer: str = "",
    ) -> None:
        assert len(labels) == len(filenames) == len(page_rows)
        for page_index, (label, filename, rows) in enumerate(zip(labels, filenames, page_rows), start=1):
            page_title = title if page_index == 1 else title + " — continued"
            page_subtitle = subtitle + (f" Part {page_index} of {len(page_rows)}." if len(page_rows) > 1 else "")
            entry = draw_forest_page(
                filename,
                f"Figure {label} · {kicker_domain}",
                page_title,
                page_subtitle,
                rows,
                effect_type,
                context=context,
                band_key=band_key,
                footer=footer,
            )
            entry.update(
                figure=label,
                domain=domain,
                selection_rule=selection_rule,
                caption_notes=caption_notes + (f" Part {page_index} of {len(page_rows)}." if len(page_rows) > 1 else ""),
            )
            manifest.append(entry)

    med_selected = choose_med_forest(med)
    add_pages(
        labels=["9A1", "9A2"],
        filenames=["figure9a1_medwas_forest.png", "figure9a2_medwas_forest.png"],
        page_rows=[med_selected[:20], med_selected[20:]],
        kicker_domain="Medication-wide association study",
        title="Strongest medication-class associations, paired across windows",
        subtitle="Twenty descriptors balanced by direction; each is shown at 1 and 5 years for the primary fully adjusted contrast.",
        effect_type="OR",
        context=lambda row: f"{'1-year window' if row['window'] == '1yr' else '5-year window'}|{forest_group_label(row['group'])}",
        band_key=lambda row: str(row["code"]),
        domain="MedWAS",
        selection_rule="20 strongest FDR descriptors balanced 10 positive/10 inverse, each paired at 1 and 5 years; 40 rows total",
        caption_notes="ORs model any medication-class fill; pages preserve 1-year/5-year descriptor pairs and use a log-scale null of 1.",
    )

    proc_selected = sorted(
        [row for row in proc if as_bool(row["sig_fdr"])],
        key=lambda row: (str(row["group"]), str(row["label"]), str(row["window"])),
    )
    assert len(proc_selected) == 38
    add_pages(
        labels=["9B1", "9B2"],
        filenames=["figure9b1_procwas_forest.png", "figure9b2_procwas_forest.png"],
        page_rows=[proc_selected[:19], proc_selected[19:]],
        kicker_domain="Procedure-wide association study",
        title="All FDR-significant procedure-rate associations",
        subtitle="Fully adjusted incidence-rate ratios compare AHI ≥5 with AHI <5 across the 1- and 5-year windows.",
        effect_type="IRR",
        context=lambda row: f"{'1-year window' if row['window'] == '1yr' else '5-year window'}|{forest_group_label(row['group'])}",
        band_key=lambda row: str(row["group"]),
        domain="ProcWAS",
        selection_rule="All 38 FDR-significant primary ProcWAS rows, split 19/19",
        caption_notes="IRRs model procedure rates over observed person-time; null is 1 on a log scale.",
    )

    util_presence_1 = sorted(
        [row for row in util if row["part"] == "presence" and row["window"] == "1yr" and as_bool(row["sig_fdr"])],
        key=lambda row: (str(row["combo"]), str(row["detail"]), str(row["label"])),
    )
    assert len(util_presence_1) == 26
    add_pages(
        labels=["9C1a", "9C1b"],
        filenames=["figure9c1a_utilwas_ed_presence_1yr_forest.png", "figure9c1b_utilwas_ed_presence_1yr_forest.png"],
        page_rows=[util_presence_1[:13], util_presence_1[13:]],
        kicker_domain="Emergency-department UtilWAS",
        title="ED diagnoses present during the 1-year window",
        subtitle="All FDR-significant presence estimates; odds ratios model any qualifying ED diagnosis in the full cohort.",
        effect_type="OR",
        context=lambda row: f"1-year · {'primary' if row['combo'] == 'ED_primary' else 'secondary'} ED|{forest_group_label(row['detail'])}",
        band_key=lambda row: f"{row['combo']}|{row['detail']}",
        domain="UtilWAS, ED visits only",
        selection_rule="All 26 FDR-significant 1-year ED diagnosis-presence rows, split 13/13",
        caption_notes="Presence is a full-cohort OR estimand; ED-primary and ED-secondary roles remain explicit; null is 1.",
    )

    util_presence_5_primary = sorted(
        [row for row in util if row["part"] == "presence" and row["window"] == "5yr" and row["combo"] == "ED_primary" and as_bool(row["sig_fdr"])],
        key=lambda row: (str(row["detail"]), str(row["label"])),
    )
    assert len(util_presence_5_primary) == 11
    add_pages(
        labels=["9C2"],
        filenames=["figure9c2_utilwas_ed_primary_presence_5yr_forest.png"],
        page_rows=[util_presence_5_primary],
        kicker_domain="Emergency-department UtilWAS",
        title="Primary ED diagnoses present during the 5-year window",
        subtitle="All FDR-significant presence estimates; odds ratios model any qualifying primary ED diagnosis in the full cohort.",
        effect_type="OR",
        context=lambda row: f"5-year · primary ED|{forest_group_label(row['detail'])}",
        band_key=lambda row: str(row["detail"]),
        domain="UtilWAS, ED visits only",
        selection_rule="All 11 FDR-significant 5-year ED-primary diagnosis-presence rows",
        caption_notes="Presence is a full-cohort OR estimand; null is 1 on a log scale.",
    )

    util_presence_5_secondary = sorted(
        [row for row in util if row["part"] == "presence" and row["window"] == "5yr" and row["combo"] == "ED_secondary" and as_bool(row["sig_fdr"])],
        key=lambda row: (str(row["detail"]), str(row["label"])),
    )
    assert len(util_presence_5_secondary) == 39
    add_pages(
        labels=["9C3a", "9C3b"],
        filenames=["figure9c3a_utilwas_ed_secondary_presence_5yr_forest.png", "figure9c3b_utilwas_ed_secondary_presence_5yr_forest.png"],
        page_rows=[util_presence_5_secondary[:20], util_presence_5_secondary[20:]],
        kicker_domain="Emergency-department UtilWAS",
        title="Secondary ED diagnoses present during the 5-year window",
        subtitle="All FDR-significant presence estimates; odds ratios model any qualifying secondary ED diagnosis in the full cohort.",
        effect_type="OR",
        context=lambda row: f"5-year · secondary ED|{forest_group_label(row['detail'])}",
        band_key=lambda row: str(row["detail"]),
        domain="UtilWAS, ED visits only",
        selection_rule="All 39 FDR-significant 5-year ED-secondary diagnosis-presence rows, split 20/19",
        caption_notes="Presence is a full-cohort OR estimand; null is 1 on a log scale.",
    )

    conditional_sig = [row for row in util if row["part"] == "count_among_present" and as_bool(row["sig_fdr"])]
    assert len(conditional_sig) == 1
    comparators: list[dict[str, object]] = []
    for window in ("1yr", "5yr"):
        for combo in ("ED_primary", "ED_secondary"):
            candidates = [
                row
                for row in util
                if row["part"] == "count_among_present"
                and row["window"] == window
                and row["combo"] == combo
                and not as_bool(row["sig_fdr"])
            ]
            candidates.sort(key=lambda row: float("inf") if row["p"] is None else float(row["p"]))
            comparators.extend(candidates[:2])
    conditional = sorted(
        conditional_sig + comparators,
        key=lambda row: (str(row["window"]), str(row["combo"]), not as_bool(row["sig_fdr"]), float("inf") if row["p"] is None else float(row["p"])),
    )
    assert len(conditional) == 9
    add_pages(
        labels=["9C4"],
        filenames=["figure9c4_utilwas_ed_conditional_rate_forest.png"],
        page_rows=[conditional],
        kicker_domain="Emergency-department UtilWAS",
        title="ED encounter rates among patients with the diagnosis present",
        subtitle="The sole FDR-significant conditional-rate result and two leading FDR-null comparators from each ED role × window stratum.",
        effect_type="IRR",
        context=lambda row: f"{'1-year' if row['window'] == '1yr' else '5-year'} · {'primary' if row['combo'] == 'ED_primary' else 'secondary'} ED|{forest_group_label(row['detail'])}",
        band_key=lambda row: f"{row['window']}|{row['combo']}",
        domain="UtilWAS, ED visits only",
        selection_rule="The sole FDR-significant ED conditional-rate row plus the two lowest-P FDR-null comparators from each window × diagnosis-role stratum; 9 rows total",
        caption_notes="Conditional IRRs apply only among patients with ≥1 qualifying ED encounter for the diagnosis and must not be interpreted as presence estimates; null rows use light directional triangles.",
        footer="Conditional-rate estimates are restricted to patients with ≥1 qualifying ED encounter for the listed diagnosis and are not prevalence estimates.",
    )

    q_binary = sorted(
        [row for row in qwas if row["form"] == "binary" and as_bool(row["sig_fdr"])],
        key=lambda row: (str(row["group"]), str(row["label"])),
    )
    assert len(q_binary) == 44
    add_pages(
        labels=["10A1a", "10A1b"],
        filenames=["figure10a1a_qwas_binary_forest.png", "figure10a1b_qwas_binary_forest.png"],
        page_rows=[q_binary[:22], q_binary[22:]],
        kicker_domain="Questionnaire-wide association study",
        title="All FDR-significant binary questionnaire associations",
        subtitle="Odds ratios compare AHI ≥5 with AHI <5 using feature-specific complete cases in the fully adjusted model.",
        effect_type="OR",
        context=lambda row: str(row["group"]),
        band_key=lambda row: str(row["group"]),
        domain="QWAS",
        selection_rule="All 44 FDR-significant binary QWAS rows, split 22/22",
        caption_notes="ORs use feature-specific complete cases; asterisks identify referral-driving ascertainment-sensitive symptoms; null is 1.",
        footer="* Referral-driving symptom; large estimates may partly reflect the sleep-study referral pathway.",
    )

    q_continuous = sorted(
        [row for row in qwas if row["form"] == "continuous" and as_bool(row["sig_fdr"])],
        key=lambda row: (str(row["group"]), str(row["label"])),
    )
    assert len(q_continuous) == 37
    add_pages(
        labels=["10A2a", "10A2b"],
        filenames=["figure10a2a_qwas_continuous_forest.png", "figure10a2b_qwas_continuous_forest.png"],
        page_rows=[q_continuous[:19], q_continuous[19:]],
        kicker_domain="Questionnaire-wide association study",
        title="All FDR-significant continuous questionnaire associations",
        subtitle="β values are adjusted differences in rank-inverse-normal scores (approximately SD units) for AHI ≥5 versus AHI <5.",
        effect_type="β",
        context=lambda row: str(row["group"]),
        band_key=lambda row: str(row["group"]),
        domain="QWAS",
        selection_rule="All 37 FDR-significant continuous QWAS rows, split 19/18",
        caption_notes="Beta is in rank-inverse-normal SD units; asterisks identify referral-driving ascertainment-sensitive symptoms; higher FOSQ scores indicate better function.",
        footer="* Referral-driving symptom; large estimates may partly reflect the sleep-study referral pathway. Higher FOSQ scores indicate better function.",
    )

    beh_binary = sorted(
        [row for row in beh if row["effect_type"] == "OR"],
        key=lambda row: ({"1yr": 0, "5yr": 1, "lifetime": 2}[str(row["window"])], str(row["label"])),
    )
    assert len(beh_binary) == 12
    add_pages(
        labels=["10B1"],
        filenames=["figure10b1_behwas_binary_forest.png"],
        page_rows=[beh_binary],
        kicker_domain="Behavior-wide association study",
        title="All primary binary behavior associations",
        subtitle="Odds ratios compare AHI ≥5 with AHI <5; FDR-null estimates are retained to show the complete prespecified item set.",
        effect_type="OR",
        context=lambda row: str(row["group"]),
        band_key=lambda row: str(row["group"]),
        domain="BehWAS",
        selection_rule="All 12 primary binary BehWAS rows across 1-year, 5-year, and undated/lifetime windows",
        caption_notes="Medication adherence is among users; lifetime rows are descriptive; BehWAS is preliminary; light triangles are FDR-null but retain effect direction.",
        footer="Medication adherence is evaluated among medication users. Undated/lifetime outcomes are descriptive, and BehWAS is preliminary.",
    )

    beh_continuous = sorted(
        [row for row in beh if row["effect_type"] == "beta"],
        key=lambda row: ({"1yr": 0, "5yr": 1, "lifetime": 2}[str(row["window"])], str(row["label"])),
    )
    assert len(beh_continuous) == 8
    add_pages(
        labels=["10B2"],
        filenames=["figure10b2_behwas_continuous_forest.png"],
        page_rows=[beh_continuous],
        kicker_domain="Behavior-wide association study",
        title="All primary continuous behavior associations",
        subtitle="β values are adjusted differences in rank-inverse-normal scores; FDR-null estimates are retained for complete item coverage.",
        effect_type="β",
        context=lambda row: str(row["group"]),
        band_key=lambda row: str(row["group"]),
        domain="BehWAS",
        selection_rule="All 8 primary continuous BehWAS rows across 1-year, 5-year, and undated/lifetime windows",
        caption_notes="Medication PDC is among users; lifetime rows are descriptive; BehWAS is preliminary; light triangles are FDR-null but retain effect direction.",
        footer="Medication PDC is evaluated among medication users. Undated/lifetime outcomes are descriptive, and BehWAS is preliminary.",
    )
    return manifest


def main() -> None:
    legacy_outputs = [
        "figure5_medwas_procwas_manhattan.png",
        "figure6_utilwas_ed_qwas_manhattan.png",
        "figure6a_utilwas_ed_presence_manhattan.png",
        "figure6b_utilwas_ed_conditional_manhattan.png",
        "figure7_medwas_procwas_volcano.png",
        "figure8_utilwas_ed_qwas_volcano.png",
        "figure8a_utilwas_ed_presence_volcano.png",
        "figure8b_utilwas_ed_conditional_volcano.png",
        "figure9a_medwas_forest.png",
        "figure9b_procwas_forest.png",
        "figure9c1_utilwas_ed_presence_1yr_forest.png",
        "figure9c3_utilwas_ed_secondary_presence_5yr_forest.png",
        "figure10a1_qwas_binary_forest.png",
        "figure10a2_qwas_continuous_forest.png",
    ]
    resolved_out = OUT.resolve()
    for filename in legacy_outputs:
        path = (OUT / filename).resolve()
        assert path.parent == resolved_out
        if path.exists():
            path.unlink()
    data = load_data()
    expected = {"med": 395, "proc": 92, "util": 540, "qwas": 132, "beh": 20}
    observed = {key: len(value) for key, value in data.items()}
    assert observed == expected, f"Unexpected primary row inventory: {observed}"
    manifest = build_manhattan_and_volcano(data) + build_forests(data)
    for entry in manifest:
        path = OUT / str(entry["file"])
        with Image.open(path) as rendered:
            assert rendered.size == (entry["width"], entry["height"])
            assert rendered.mode == "RGB"
            assert rendered.height / rendered.width <= 1.35
        for required in ("figure", "domain", "effect_type", "rows", "selection_rule", "caption_notes"):
            assert required in entry, f"Manifest field {required!r} missing for {entry['file']}"
    output = {
        "source_primary_rows": observed,
        "dpi": 300,
        "word_width_inches": WIDTH / 300,
        "assets": manifest,
    }
    manifest_path = OUT / "figure_manifest_260905.json"
    manifest_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
