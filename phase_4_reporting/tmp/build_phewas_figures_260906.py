from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path
from typing import Callable

from PIL import Image, ImageColor, ImageDraw


ROOT = Path(r"C:\Users\natha\Projects\work\phewas")
OUT = ROOT / "tmp" / "phewas_260906_assets"
OUT.mkdir(parents=True, exist_ok=True)

LEGACY_SCRIPT = ROOT / "tmp" / "build_phewas_figures_260905.py"
SPEC = importlib.util.spec_from_file_location("phewas_260905", LEGACY_SCRIPT)
assert SPEC is not None and SPEC.loader is not None
base = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(base)
base.OUT = OUT

WIDTH = base.WIDTH
DPI = base.DPI
BG = base.BG
PANEL_BG = base.PANEL_BG
INK = base.INK
MID_INK = base.MID_INK
GRID = base.GRID
TEAL = base.TEAL
TEAL_LIGHT = base.TEAL_LIGHT
PURPLE = base.PURPLE
PURPLE_LIGHT = base.PURPLE_LIGHT
OUTLINE = base.OUTLINE
MUTED = base.MUTED
WHITE = base.WHITE
BAND_A = base.BAND_A
BAND_B = base.BAND_B
ORANGE = "#D86A2B"

SAFE_INPUTS = {
    "results/icd_analysis_v2_results/summary/phewas_results_combined.csv",
    "results/incwas_results/summary/cox_results_combined.csv",
    "results/incwas_results/phenotype_exposure/website_exports/octant_cluster_distributions.csv",
    "results/incwas_results/phenotype_exposure/website_exports/octant_cif_metadata.csv",
    "website/public/data/phenotype-survival/phecode/428-2.json",
}
FORBIDDEN_PARTS = (
    "data/raw",
    "/phi/",
    "patient_records",
    "_identified",
    "assignments",
    "shared_mrns",
    "cohort.parquet",
    "/.ssh/",
    "/.aws/",
)


def safe_path(relative: str) -> Path:
    normalized = relative.replace("\\", "/")
    lower = f"/{normalized.lower()}/"
    assert normalized in SAFE_INPUTS, f"Input is not allow-listed: {relative}"
    assert not any(part in lower for part in FORBIDDEN_PARTS), f"Unsafe path refused: {relative}"
    return ROOT / normalized


def read_csv(relative: str) -> list[dict[str, str]]:
    with safe_path(relative).open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def read_json(relative: str) -> dict[str, object]:
    return json.loads(safe_path(relative).read_text(encoding="utf-8"))


def is_significant(row: dict[str, object]) -> bool:
    return base.as_bool(row.get("sig_fdr")) or base.as_bool(row.get("sig_bon"))


def marker_style(row: dict[str, object]) -> tuple[int, str, str | None, int]:
    direction = 1 if float(row["signed"]) >= 0 else -1
    if is_significant(row):
        return direction, TEAL if direction > 0 else PURPLE, OUTLINE, 2
    return direction, TEAL_LIGHT if direction > 0 else PURPLE_LIGHT, "#879594", 1


def choose_key_labels(rows: list[dict[str, object]], max_labels: int = 20) -> list[dict[str, object]]:
    eligible = [row for row in rows if is_significant(row)]
    by_direction: dict[int, list[dict[str, object]]] = {1: [], -1: []}
    for row in eligible:
        by_direction[1 if float(row["signed"]) >= 0 else -1].append(row)
    for values in by_direction.values():
        values.sort(
            key=lambda row: (
                float("inf") if row.get("p") is None else float(row["p"]),
                -float(row["magnitude"]),
                str(row["label"]).lower(),
            )
        )
    half = max_labels // 2
    selected = by_direction[1][:half] + by_direction[-1][:half]
    remaining = sorted(
        [row for row in eligible if row not in selected],
        key=lambda row: (
            float("inf") if row.get("p") is None else float(row["p"]),
            -float(row["magnitude"]),
            str(row["label"]).lower(),
        ),
    )
    selected.extend(remaining[: max_labels - len(selected)])
    selected.sort(
        key=lambda row: (
            float("inf") if row.get("p") is None else float(row["p"]),
            str(row["label"]).lower(),
        )
    )
    assert len(selected) <= max_labels
    assert all(is_significant(row) for row in selected)
    return selected


def annotate_point(
    draw: ImageDraw.ImageDraw,
    point: tuple[int, int],
    label: str,
    plot_box: tuple[int, int, int, int],
    occupied: list[tuple[int, int, int, int]],
    index: int,
) -> None:
    label_font = base.font(15, bold=True)
    max_width = 190
    lines = base.wrap_to_width(draw, base.shortened(label, 52), label_font, max_width)[:2]
    line_height = 18
    text_width = max(draw.textbbox((0, 0), line, font=label_font)[2] for line in lines)
    box_w = text_width + 12
    box_h = len(lines) * line_height + 8
    px, py = point
    x0, y0, x1, y1 = plot_box
    candidates: list[tuple[float, float]] = []
    for distance in (13, 38, 72, 112):
        candidates.extend(
            [
                (px + distance, py - box_h - 5),
                (px - box_w - distance, py - box_h - 5),
                (px + distance, py + 8),
                (px - box_w - distance, py + 8),
            ]
        )
    grid_step_x = max(150, int((x1 - x0) / 9))
    grid_step_y = max(48, box_h + 6)
    for gy in range(y0 + 5, max(y0 + 6, y1 - box_h), grid_step_y):
        for gx in range(x0 + 5, max(x0 + 6, x1 - box_w), grid_step_x):
            candidates.append((gx, gy))
    candidates.sort(key=lambda candidate: (candidate[0] - px) ** 2 + (candidate[1] - py) ** 2)
    chosen: tuple[int, int, int, int] | None = None
    for bx, by in candidates:
        bx = min(max(bx, x0 + 2), x1 - box_w - 2)
        by = min(max(by, y0 + 2), y1 - box_h - 2)
        box = (int(bx), int(by), int(bx + box_w), int(by + box_h))
        if not any(base.overlap(box, old, pad=3) for old in occupied):
            chosen = box
            break
    if chosen is None:
        lane_x = x0 + 4 + (index % 5) * max(1, (x1 - x0 - box_w - 8) // 4)
        lane_y = y0 + 4 + (index // 5) * (box_h + 5)
        chosen = (int(lane_x), int(lane_y), int(lane_x + box_w), int(lane_y + box_h))
    occupied.append(chosen)
    bx0, by0, bx1, by1 = chosen
    anchor_x = bx0 if px < (bx0 + bx1) / 2 else bx1
    anchor_y = min(max(py, by0 + 3), by1 - 3)
    draw.line((px, py, anchor_x, anchor_y), fill="#627978", width=2)
    draw.rounded_rectangle(chosen, radius=4, fill=WHITE, outline="#9CB0AD", width=1)
    for line_no, line in enumerate(lines):
        draw.text((bx0 + 6, by0 + 3 + line_no * line_height), line, fill=INK, font=label_font)


def draw_header(
    image: Image.Image,
    kicker: str,
    title: str,
    subtitle: str,
    *,
    legend: bool = True,
) -> int:
    draw = ImageDraw.Draw(image)
    draw.text((70, 52), kicker.upper(), fill=TEAL, font=base.font(21, bold=True))
    draw.text((70, 86), title, fill=INK, font=base.font(48, serif=True))
    draw.text((72, 151), subtitle, fill=MID_INK, font=base.font(23))
    draw.line((70, 194, WIDTH - 70, 194), fill="#B8CFCA", width=2)
    if not legend:
        return 225
    y = 239
    base.triangle(draw, 88, y, 11, 1, TEAL, OUTLINE, 2)
    draw.text((108, y - 14), "positive / higher in AHI ≥5", fill=INK, font=base.font(20))
    base.triangle(draw, 425, y, 11, -1, PURPLE, OUTLINE, 2)
    draw.text((445, y - 14), "inverse / lower in AHI ≥5", fill=INK, font=base.font(20))
    base.triangle(draw, 805, y, 9, 1, TEAL, OUTLINE, 2)
    draw.text((824, y - 14), "FDR/Bonferroni significant", fill=INK, font=base.font(20))
    base.triangle(draw, 1170, y, 9, 1, TEAL_LIGHT, "#879594", 1)
    draw.text((1189, y - 14), "multiplicity-null", fill=INK, font=base.font(20))
    base.triangle(draw, 1480, y, 6, 1, MUTED)
    base.triangle(draw, 1530, y, 10, 1, MUTED)
    base.triangle(draw, 1590, y, 14, 1, MUTED)
    draw.text((1620, y - 14), "larger = greater |effect|", fill=INK, font=base.font(19))
    return 285


base.marker_style = marker_style
base.choose_key_labels = choose_key_labels
base.annotate_point = annotate_point
base.draw_header = draw_header


def ratio_row(
    *,
    effect: object,
    ci_low: object,
    ci_high: object,
    p: object,
    sig_fdr: object,
    sig_bon: object,
    label: str,
    group: str,
    code: str,
) -> dict[str, object] | None:
    row = base.enrich(
        {
            "effect_type": "OR",
            "effect": effect,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "p": p,
            "sig_fdr": sig_fdr,
            "sig_bon": sig_bon,
            "label": label,
            "group": group,
            "code": code,
        }
    )
    return row


def load_disease_data() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    prevalence: list[dict[str, object]] = []
    for source in read_csv("results/icd_analysis_v2_results/summary/phewas_results_combined.csv"):
        if source["contrast"] != "ahi_ge5" or source["model"] != "m4_bmi":
            continue
        if source["PheWAS Code"].startswith("327.3"):
            continue
        row = ratio_row(
            effect=source["OR"],
            ci_low=source["OR_low"],
            ci_high=source["OR_high"],
            p=source["p-val"],
            sig_fdr=source["sig_fdr"],
            sig_bon=source["sig_bon"],
            label=base.smart_title(source["PheWAS Name"]),
            group=base.smart_title(source["category_string"]),
            code=source["PheWAS Code"],
        )
        if row is not None:
            prevalence.append(row)
    incidence: list[dict[str, object]] = []
    for source in read_csv("results/incwas_results/summary/cox_results_combined.csv"):
        if source["contrast"] != "ahi_ge5" or source["model"] != "m4":
            continue
        if source["phecode"].startswith("327.3"):
            continue
        row = ratio_row(
            effect=source["effect"],
            ci_low=source["ci_low"],
            ci_high=source["ci_high"],
            p=source["p"],
            sig_fdr=source["sig_fdr"],
            sig_bon=source["sig_bon"],
            label=base.smart_title(source["phenotype"]),
            group=base.smart_title(source["category"]),
            code=source["phecode"],
        )
        if row is not None:
            incidence.append(row)
    assert len(prevalence) == 1779, len(prevalence)
    assert len(incidence) == 1012, len(incidence)
    return prevalence, incidence


def labeled_payload(panels: list[tuple[str, list[dict[str, object]], str]]) -> tuple[dict[str, int], dict[str, list[str]]]:
    counts: dict[str, int] = {}
    labels: dict[str, list[str]] = {}
    for panel_title, rows, _ in panels:
        chosen = choose_key_labels(rows)
        assert len(chosen) <= 20 and all(is_significant(row) for row in chosen)
        counts[panel_title] = len(chosen)
        labels[panel_title] = [str(row["label"]) for row in chosen]
    return counts, labels


def add_chart_page(
    manifest: list[dict[str, object]],
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
    boxes = base.grid_boxes(top, 2405 if footer else 2440, 1, len(panels))
    for box, (panel_title, rows, xlabel) in zip(boxes, panels):
        if chart == "manhattan":
            base.render_manhattan_panel(image, box, panel_title, rows)
        else:
            base.render_volcano_panel(image, box, panel_title, rows, xlabel)
    if footer:
        ImageDraw.Draw(image).text((72, 2435), footer, fill=MID_INK, font=base.font(21, italic=True))
    path = base.save_page(image, filename)
    label_counts, labels = labeled_payload(panels)
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
            "label_counts": label_counts,
            "labeled_outcomes": labels,
            "label_rule": "At most 20 per panel; FDR- or Bonferroni-significant rows only, selected by P value with directional balance where possible.",
        }
    )


def add_forest_pages(
    manifest: list[dict[str, object]],
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
        entry = base.draw_forest_page(
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
            selected_outcomes=[
                {
                    key: row[key]
                    for key in ("code", "label", "window", "outcome", "combo", "part")
                    if key in row
                }
                for row in rows
            ],
        )
        manifest.append(entry)


def draw_hr_forest_page(
    filename: str,
    kicker: str,
    title: str,
    subtitle: str,
    rows: list[dict[str, object]],
    context: Callable[[dict[str, object]], str],
    footer: str = "",
) -> dict[str, object]:
    row_count = len(rows)
    height = max(1500, min(2600, 650 + row_count * 78))
    image = Image.new("RGB", (WIDTH, height), BG)
    draw = ImageDraw.Draw(image)
    top = draw_header(image, kicker, title, subtitle, legend=False)
    label_left, context_x, plot_left, plot_right, stat_x = 72, 650, 950, 1515, 1545
    header_y = top + 12
    draw.text((label_left, header_y), "INCIDENT OUTCOME", fill=TEAL, font=base.font(22, bold=True))
    draw.text((context_x, header_y), "CONTEXT", fill=TEAL, font=base.font(22, bold=True))
    draw.text((plot_left, header_y), "HR (95% CI)", fill=TEAL, font=base.font(22, bold=True))
    draw.text((stat_x, header_y), "ESTIMATE (95% CI)", fill=TEAL, font=base.font(22, bold=True))
    table_top = header_y + 52
    table_bottom = height - 165
    row_height = (table_bottom - table_top) / max(row_count, 1)
    lo, hi, ticks = base.forest_ticks(rows, "OR")
    previous_group: str | None = None
    band_index = -1
    for index, row in enumerate(rows):
        group = str(row["group"])
        if group != previous_group:
            band_index += 1
            previous_group = group
        y0 = table_top + index * row_height
        y1 = table_top + (index + 1) * row_height
        if band_index % 2 == 0:
            draw.rectangle((60, y0, WIDTH - 60, y1), fill=BAND_A)
        draw.line((60, y1, WIDTH - 60, y1), fill="#E0E7E5", width=1)
    for tick_value, tick_label in ticks:
        x = plot_left + (tick_value - lo) / (hi - lo) * (plot_right - plot_left)
        is_null = abs(tick_value) < 1e-12
        draw.line((x, table_top, x, table_bottom), fill="#637674" if is_null else GRID, width=3 if is_null else 1)
        tbox = draw.textbbox((0, 0), tick_label, font=base.font(22))
        draw.text((x - (tbox[2] - tbox[0]) / 2, table_bottom + 10), tick_label, fill=MID_INK, font=base.font(22))
    label_font = base.font(28, bold=True)
    context_font = base.font(26)
    stat_font = base.font(26)
    for index, row in enumerate(rows):
        y = table_top + (index + 0.5) * row_height
        label_lines = base.wrap_to_width(draw, str(row["label"]), label_font, context_x - label_left - 30)
        for line_no, line in enumerate(label_lines):
            draw.text((label_left, y - len(label_lines) * 29 / 2 + line_no * 29), line, fill=INK, font=label_font)
        ctx_lines: list[str] = []
        for part in [part.strip() for part in context(row).split("|") if part.strip()]:
            ctx_lines.extend(base.wrap_to_width(draw, part, context_font, plot_left - context_x - 24))
        for line_no, line in enumerate(ctx_lines):
            draw.text((context_x, y - len(ctx_lines) * 29 / 2 + line_no * 29), line, fill=MID_INK, font=context_font)
        x_low = plot_left + (float(row["signed_low"]) - lo) / (hi - lo) * (plot_right - plot_left)
        x_mid = plot_left + (float(row["signed"]) - lo) / (hi - lo) * (plot_right - plot_left)
        x_high = plot_left + (float(row["signed_high"]) - lo) / (hi - lo) * (plot_right - plot_left)
        significant = is_significant(row)
        direction = 1 if float(row["signed"]) >= 0 else -1
        fill = (TEAL if direction > 0 else PURPLE) if significant else (TEAL_LIGHT if direction > 0 else PURPLE_LIGHT)
        draw.line((x_low, y, x_high, y), fill="#536967" if significant else "#9CA9A7", width=4 if significant else 3)
        draw.line((x_low, y - 7, x_low, y + 7), fill="#536967", width=2)
        draw.line((x_high, y - 7, x_high, y + 7), fill="#536967", width=2)
        base.triangle(draw, x_mid, y, 10, direction, fill, OUTLINE if significant else "#7F8E8C", 2 if significant else 1)
        estimate = f"{float(row['effect']):.2f} ({float(row['ci_low']):.2f}–{float(row['ci_high']):.2f})"
        if significant:
            estimate += "  FDR"
        draw.text((stat_x, y - 15), estimate, fill=INK if significant else MID_INK, font=stat_font)
    axis_label = "Hazard ratio (log scale)"
    axis_box = draw.textbbox((0, 0), axis_label, font=base.font(24, bold=True))
    draw.text(((plot_left + plot_right - (axis_box[2] - axis_box[0])) / 2, table_bottom + 52), axis_label, fill=INK, font=base.font(24, bold=True))
    if footer:
        draw.text((72, height - 58), footer, fill=MID_INK, font=base.font(21, italic=True))
    path = base.save_page(image, filename)
    return {"file": path.name, "width": WIDTH, "height": height, "rows": row_count, "effect_type": "HR"}


def build_disease_figures(
    prevalence: list[dict[str, object]],
    incidence: list[dict[str, object]],
) -> list[dict[str, object]]:
    manifest: list[dict[str, object]] = []
    prevalence_panel = [("Prevalent phenotypes", prevalence, "log2(OR)")]
    incidence_panel = [("Incident phenotypes", incidence, "log2(HR)")]
    add_chart_page(
        manifest,
        figure="1A",
        filename="figure1a_prevalence_manhattan.png",
        kicker_domain="Prevalence phenotype-wide association study",
        title="Prevalent disease associations across Phecode systems",
        subtitle="Fully adjusted m4_bmi odds ratios compare AHI ≥5 with AHI <5; shaded bands delineate disease domains.",
        panels=prevalence_panel,
        chart="manhattan",
        domain="Prevalence PheDAS/PheWAS",
        effect_type="OR",
        selection_rule=f"All {len(prevalence):,} primary prevalence rows (AHI ≥5 versus <5; m4_bmi), excluding Phecode 327.3 and its subcodes",
        caption_notes="Y is -log10(P), capped at 50; shaded bands are Phecode categories; size reflects absolute log2(OR); labels are limited to 20 multiplicity-significant outcomes.",
    )
    add_chart_page(
        manifest,
        figure="1B",
        filename="figure1b_prevalence_volcano.png",
        kicker_domain="Prevalence phenotype-wide association study",
        title="Direction, magnitude, and statistical strength of prevalent disease associations",
        subtitle="Vertical position is -log10(P), capped at 50; horizontal position is signed log2(OR).",
        panels=prevalence_panel,
        chart="volcano",
        domain="Prevalence PheDAS/PheWAS",
        effect_type="OR",
        selection_rule=f"All {len(prevalence):,} primary prevalence rows (AHI ≥5 versus <5; m4_bmi), excluding Phecode 327.3 and its subcodes",
        caption_notes="Point size reflects absolute log2(OR); labels are limited to 20 FDR- or Bonferroni-significant outcomes, selected with directional balance.",
    )
    prevalence_fdr = [row for row in prevalence if base.as_bool(row["sig_fdr"])]
    positive = sorted(
        [row for row in prevalence_fdr if float(row["signed"]) >= 0],
        key=lambda row: (float(row["p"]), -float(row["magnitude"])),
    )[:20]
    inverse = sorted(
        [row for row in prevalence_fdr if float(row["signed"]) < 0],
        key=lambda row: (float(row["p"]), -float(row["magnitude"])),
    )[:20]
    assert len(positive) == len(inverse) == 20
    for label, filename, rows, direction_text in [
        ("1C1", "figure1c1_prevalence_forest_positive.png", positive, "higher prevalence"),
        ("1C2", "figure1c2_prevalence_forest_inverse.png", inverse, "lower prevalence"),
    ]:
        entry = base.draw_forest_page(
            filename,
            f"Figure {label} · Prevalence phenotype-wide association study",
            f"Strongest FDR-significant associations with {direction_text}",
            "Twenty outcomes selected by P value from the fully adjusted primary prevalence contrast.",
            rows,
            "OR",
            context=lambda row: f"Phecode {row['code']}|{base.forest_group_label(row['group'])}",
            band_key=lambda row: str(row["group"]),
        )
        entry.update(
            figure=label,
            domain="Prevalence PheDAS/PheWAS",
            selection_rule="40 FDR-significant outcomes balanced 20 positive and 20 inverse; this page contains the " + direction_text + " set",
            caption_notes="ORs are fully adjusted m4_bmi estimates; null is 1 on a log scale.",
            selected_outcomes=[{"code": row["code"], "label": row["label"]} for row in rows],
        )
        manifest.append(entry)

    add_chart_page(
        manifest,
        figure="2A",
        filename="figure2a_incidence_manhattan.png",
        kicker_domain="Incidence phenotype-wide association study",
        title="Incident disease associations across Phecode systems",
        subtitle="Fully adjusted m4 cause-specific hazard ratios compare AHI ≥5 with AHI <5; shaded bands delineate disease domains.",
        panels=incidence_panel,
        chart="manhattan",
        domain="Incidence PheDAS/PheWAS",
        effect_type="HR",
        selection_rule=f"All {len(incidence):,} primary incidence rows (AHI ≥5 versus <5; m4), excluding Phecode 327.3 and its subcodes",
        caption_notes="Y is -log10(P), capped at 50; shaded bands are Phecode categories; size reflects absolute log2(HR); labels are limited to 20 multiplicity-significant outcomes.",
    )
    add_chart_page(
        manifest,
        figure="2B",
        filename="figure2b_incidence_volcano.png",
        kicker_domain="Incidence phenotype-wide association study",
        title="Direction, magnitude, and statistical strength of incident disease associations",
        subtitle="Vertical position is -log10(P), capped at 50; horizontal position is signed log2(HR).",
        panels=incidence_panel,
        chart="volcano",
        domain="Incidence PheDAS/PheWAS",
        effect_type="HR",
        selection_rule=f"All {len(incidence):,} primary incidence rows (AHI ≥5 versus <5; m4), excluding Phecode 327.3 and its subcodes",
        caption_notes="Point size reflects absolute log2(HR); labels are limited to 20 FDR- or Bonferroni-significant outcomes, selected with directional balance.",
    )
    incidence_selected = [row for row in incidence if base.as_bool(row["sig_fdr"])]
    assert len(incidence_selected) == 27
    af = [row for row in incidence if str(row["code"]) == "427.2"]
    assert len(af) == 1 and not base.as_bool(af[0]["sig_fdr"])
    incidence_selected = sorted(
        incidence_selected + af,
        key=lambda row: (str(row["group"]).lower(), str(row["label"]).lower()),
    )
    assert len(incidence_selected) == 28
    pages = [incidence_selected[:14], incidence_selected[14:]]
    for page_index, (label, filename, rows) in enumerate(
        zip(
            ["2C1", "2C2"],
            ["figure2c1_incidence_forest.png", "figure2c2_incidence_forest.png"],
            pages,
        ),
        start=1,
    ):
        entry = draw_hr_forest_page(
            filename,
            f"Figure {label} · Incidence phenotype-wide association study",
            "FDR-significant incident disease associations" + ("" if page_index == 1 else " — continued"),
            f"All 27 FDR-significant outcomes plus prespecified atrial fibrillation/flutter (Phecode 427.2). Part {page_index} of 2.",
            rows,
            context=lambda row: f"Phecode {row['code']}|{base.forest_group_label(row['group'])}",
            footer="Atrial fibrillation/flutter was prespecified for clinical context and was not FDR significant.",
        )
        entry.update(
            figure=label,
            domain="Incidence PheDAS/PheWAS",
            selection_rule="All 27 FDR-significant primary incidence rows plus atrial fibrillation/flutter (Phecode 427.2), split 14/14, excluding Phecode 327.3 and its subcodes",
            caption_notes="Cause-specific HRs are fully adjusted m4 estimates; atrial fibrillation/flutter is the single FDR-null comparator; null is 1 on a log scale.",
            selected_outcomes=[
                {
                    "code": row["code"],
                    "label": row["label"],
                    "sig_fdr": base.as_bool(row["sig_fdr"]),
                }
                for row in rows
            ],
        )
        manifest.append(entry)
    return manifest


def build_figures_5_to_8(data: dict[str, list[dict[str, object]]]) -> list[dict[str, object]]:
    manifest: list[dict[str, object]] = []
    med = [row for row in data["med"] if row["window"] == "5yr"]
    proc = [row for row in data["proc"] if row["window"] == "5yr"]
    util = [row for row in data["util"] if row["window"] == "5yr"]
    qwas = data["qwas"]
    assert len(med) == 218 and len(proc) == 67
    med_panel = [("5-year medication fills", med, "log2(OR)")]
    proc_panel = [("5-year procedure rates", proc, "log2(IRR)")]
    for figure, filename, title, panels, chart, domain, effect, rule, notes, kicker in [
        ("5A", "figure5a_medwas_manhattan.png", "Five-year medication associations across therapeutic subdomains", med_panel, "manhattan", "MedWAS", "OR", "All 218 five-year primary MedWAS rows", "Shaded bands are medication categories; size is absolute log2(OR).", "Medication-wide association study"),
        ("5B", "figure5b_procwas_manhattan.png", "Five-year procedure-rate associations across body systems", proc_panel, "manhattan", "ProcWAS", "IRR", "All 67 five-year primary ProcWAS rows", "Shaded bands are procedure body systems; size is absolute log2(IRR).", "Procedure-wide association study"),
        ("7A", "figure7a_medwas_volcano.png", "Direction, magnitude, and strength of five-year medication associations", med_panel, "volcano", "MedWAS", "OR", "All 218 five-year primary MedWAS rows", "X is log2(OR); size is absolute log2(OR).", "Medication-wide association study"),
        ("7B", "figure7b_procwas_volcano.png", "Direction, magnitude, and strength of five-year procedure-rate associations", proc_panel, "volcano", "ProcWAS", "IRR", "All 67 five-year primary ProcWAS rows", "X is log2(IRR); size is absolute log2(IRR).", "Procedure-wide association study"),
    ]:
        add_chart_page(
            manifest,
            figure=figure,
            filename=filename,
            kicker_domain=kicker,
            title=title,
            subtitle="Fully adjusted m4 estimates compare AHI ≥5 with AHI <5; only the five-year window is displayed in the main figure.",
            panels=panels,
            chart=chart,
            domain=domain,
            effect_type=effect,
            selection_rule=rule,
            caption_notes=notes + " Labels are limited to 20 FDR- or Bonferroni-significant outcomes; both 1- and 5-year estimates remain in the supplemental atlas.",
        )

    def util_panels(part: str) -> list[tuple[str, list[dict[str, object]], str]]:
        panels: list[tuple[str, list[dict[str, object]], str]] = []
        for combo in ("ED_primary", "ED_secondary"):
            rows = [
                {**row, "group": row["detail"]}
                for row in util
                if row["part"] == part and row["combo"] == combo
            ]
            panels.append(
                (
                    "Primary ED diagnoses" if combo == "ED_primary" else "Secondary ED diagnoses",
                    rows,
                    "log2(OR)" if part == "presence" else "log2(IRR)",
                )
            )
        return panels

    q_panels = [
        ("Binary questionnaire items", [row for row in qwas if row["form"] == "binary"], "log2(OR)"),
        ("Continuous questionnaire items", [row for row in qwas if row["form"] == "continuous"], "β (rank-inverse-normal SD)"),
    ]
    q_footer = "* Referral-driving questionnaire item; interpretation is ascertainment-sensitive."
    chart_specs = [
        ("6A", "figure6a_utilwas_ed_presence_5yr_manhattan.png", "Five-year presence of ED diagnoses", util_panels("presence"), "manhattan", "UtilWAS, ED visits only", "OR", "All five-year ED-primary and ED-secondary diagnosis-presence rows", "Presence ORs model any qualifying ED diagnosis in the full cohort; ED role remains explicit.", "Emergency-department utilization-wide association study", ""),
        ("6B", "figure6b_utilwas_ed_conditional_5yr_manhattan.png", "Five-year ED encounter rates among patients with each diagnosis", util_panels("count_among_present"), "manhattan", "UtilWAS, ED visits only", "IRR", "All five-year ED-primary and ED-secondary conditional-rate rows", "Conditional IRRs apply among patients with at least one qualifying ED encounter for that diagnosis.", "Emergency-department utilization-wide association study", ""),
        ("6C", "figure6c_qwas_manhattan.png", "Patient-reported associations across symptom domains", q_panels, "manhattan", "QWAS", ["OR", "beta"], "All 83 binary and 49 continuous primary QWAS rows", "Binary and continuous forms are separate; shaded bands identify questionnaire domains.", "Questionnaire-wide association study", q_footer),
        ("8A", "figure8a_utilwas_ed_presence_5yr_volcano.png", "Direction and strength of five-year ED diagnosis-presence associations", util_panels("presence"), "volcano", "UtilWAS, ED visits only", "OR", "All five-year ED-primary and ED-secondary diagnosis-presence rows", "Presence ORs model any qualifying ED diagnosis in the full cohort; ED role remains explicit.", "Emergency-department utilization-wide association study", ""),
        ("8B", "figure8b_utilwas_ed_conditional_5yr_volcano.png", "Direction and strength of five-year conditional ED encounter rates", util_panels("count_among_present"), "volcano", "UtilWAS, ED visits only", "IRR", "All five-year ED-primary and ED-secondary conditional-rate rows", "Conditional IRRs apply among patients with at least one qualifying ED encounter for that diagnosis.", "Emergency-department utilization-wide association study", ""),
        ("8C", "figure8c_qwas_volcano.png", "Direction, magnitude, and strength of patient-reported associations", q_panels, "volcano", "QWAS", ["OR", "beta"], "All 83 binary and 49 continuous primary QWAS rows", "Binary x is log2(OR); continuous x is rank-inverse-normal beta.", "Questionnaire-wide association study", q_footer),
    ]
    for figure, filename, title, panels, chart, domain, effect, rule, notes, kicker, footer in chart_specs:
        subtitle = (
            "Fully adjusted m4 estimates compare AHI ≥5 with AHI <5; only five-year ED outcomes are displayed in the main figure."
            if domain.startswith("UtilWAS")
            else "Binary and continuous forms are displayed separately; estimates compare AHI ≥5 with AHI <5."
        )
        add_chart_page(
            manifest,
            figure=figure,
            filename=filename,
            kicker_domain=kicker,
            title=title,
            subtitle=subtitle,
            panels=panels,
            chart=chart,
            domain=domain,
            effect_type=effect,
            selection_rule=rule,
            caption_notes=notes + " Labels are limited to 20 FDR- or Bonferroni-significant outcomes; both 1- and 5-year ED estimates remain in the supplemental atlas." if domain.startswith("UtilWAS") else notes + " Labels are limited to 20 FDR- or Bonferroni-significant outcomes.",
            footer=footer,
        )
    order = {"5A": 0, "5B": 1, "6A": 2, "6B": 3, "6C": 4, "7A": 5, "7B": 6, "8A": 7, "8B": 8, "8C": 9}
    manifest.sort(key=lambda entry: order[str(entry["figure"])])
    return manifest


def build_figures_9_to_10(data: dict[str, list[dict[str, object]]]) -> list[dict[str, object]]:
    manifest: list[dict[str, object]] = []
    med = [row for row in data["med"] if row["window"] == "5yr"]
    proc = [row for row in data["proc"] if row["window"] == "5yr"]
    util = [row for row in data["util"] if row["window"] == "5yr"]
    qwas = data["qwas"]
    beh = [row for row in data["beh"] if row["window"] in {"5yr", "lifetime"}]

    med_fdr = [row for row in med if base.as_bool(row["sig_fdr"])]
    med_positive = sorted(
        [row for row in med_fdr if float(row["signed"]) >= 0],
        key=lambda row: (float(row["p"]), -float(row["magnitude"])),
    )[:20]
    med_inverse = sorted(
        [row for row in med_fdr if float(row["signed"]) < 0],
        key=lambda row: (float(row["p"]), -float(row["magnitude"])),
    )[:20]
    assert len(med_positive) == len(med_inverse) == 20
    add_forest_pages(
        manifest,
        labels=["9A1", "9A2"],
        filenames=["figure9a1_medwas_5yr_forest_positive.png", "figure9a2_medwas_5yr_forest_inverse.png"],
        page_rows=[med_positive, med_inverse],
        kicker_domain="Medication-wide association study",
        title="Strongest FDR-significant five-year medication-class associations",
        subtitle="Forty medication descriptors balanced by direction (20 positive and 20 inverse) in the fully adjusted primary contrast.",
        effect_type="OR",
        context=lambda row: f"5-year window|{base.forest_group_label(row['group'])}",
        band_key=lambda row: str(row["group"]),
        domain="MedWAS",
        selection_rule="40 strongest five-year FDR-significant MedWAS rows balanced 20 positive/20 inverse, split by direction",
        caption_notes="ORs model any medication-class fill; only the five-year window is shown; both windows remain in the supplemental atlas.",
    )

    proc_fdr = sorted(
        [row for row in proc if base.as_bool(row["sig_fdr"])],
        key=lambda row: (str(row["group"]).lower(), str(row["label"]).lower()),
    )
    assert len(proc_fdr) == 28
    add_forest_pages(
        manifest,
        labels=["9B1", "9B2"],
        filenames=["figure9b1_procwas_5yr_forest.png", "figure9b2_procwas_5yr_forest.png"],
        page_rows=[proc_fdr[:14], proc_fdr[14:]],
        kicker_domain="Procedure-wide association study",
        title="All FDR-significant five-year procedure-rate associations",
        subtitle="Fully adjusted incidence-rate ratios compare AHI ≥5 with AHI <5 in the five-year window.",
        effect_type="IRR",
        context=lambda row: f"5-year window|{base.forest_group_label(row['group'])}",
        band_key=lambda row: str(row["group"]),
        domain="ProcWAS",
        selection_rule="All 28 five-year FDR-significant primary ProcWAS rows, split 14/14",
        caption_notes="IRRs model procedure rates over observed person-time; only five-year results are shown; both windows remain in the supplemental atlas.",
    )

    util_primary = sorted(
        [
            row
            for row in util
            if row["part"] == "presence" and row["combo"] == "ED_primary" and base.as_bool(row["sig_fdr"])
        ],
        key=lambda row: (str(row["detail"]).lower(), str(row["label"]).lower()),
    )
    assert len(util_primary) == 11
    add_forest_pages(
        manifest,
        labels=["9C1"],
        filenames=["figure9c1_utilwas_ed_primary_presence_5yr_forest.png"],
        page_rows=[util_primary],
        kicker_domain="Emergency-department UtilWAS",
        title="FDR-significant primary ED diagnoses present during the five-year window",
        subtitle="Odds ratios model any qualifying primary ED diagnosis in the full cohort.",
        effect_type="OR",
        context=lambda row: f"5-year · primary ED|{base.forest_group_label(row['detail'])}",
        band_key=lambda row: str(row["detail"]),
        domain="UtilWAS, ED visits only",
        selection_rule="All 11 five-year FDR-significant ED-primary diagnosis-presence rows",
        caption_notes="Presence is a full-cohort estimand; ED-primary role remains explicit; both windows remain in the supplemental atlas.",
    )

    util_secondary = sorted(
        [
            row
            for row in util
            if row["part"] == "presence" and row["combo"] == "ED_secondary" and base.as_bool(row["sig_fdr"])
        ],
        key=lambda row: (str(row["detail"]).lower(), str(row["label"]).lower()),
    )
    assert len(util_secondary) == 39
    add_forest_pages(
        manifest,
        labels=["9C2a", "9C2b"],
        filenames=["figure9c2a_utilwas_ed_secondary_presence_5yr_forest.png", "figure9c2b_utilwas_ed_secondary_presence_5yr_forest.png"],
        page_rows=[util_secondary[:20], util_secondary[20:]],
        kicker_domain="Emergency-department UtilWAS",
        title="FDR-significant secondary ED diagnoses present during the five-year window",
        subtitle="Odds ratios model any qualifying secondary ED diagnosis in the full cohort.",
        effect_type="OR",
        context=lambda row: f"5-year · secondary ED|{base.forest_group_label(row['detail'])}",
        band_key=lambda row: str(row["detail"]),
        domain="UtilWAS, ED visits only",
        selection_rule="All 39 five-year FDR-significant ED-secondary diagnosis-presence rows, split 20/19",
        caption_notes="Presence is a full-cohort estimand; ED-secondary role remains explicit; both windows remain in the supplemental atlas.",
    )

    conditional_sig = [
        row
        for row in util
        if row["part"] == "count_among_present" and base.as_bool(row["sig_fdr"])
    ]
    assert len(conditional_sig) == 1
    conditional_null: list[dict[str, object]] = []
    for combo in ("ED_primary", "ED_secondary"):
        candidates = [
            row
            for row in util
            if row["part"] == "count_among_present"
            and row["combo"] == combo
            and not base.as_bool(row["sig_fdr"])
        ]
        candidates.sort(key=lambda row: (float(row["p"]), -float(row["magnitude"])))
        conditional_null.extend(candidates[:2])
    conditional = sorted(
        conditional_sig + conditional_null,
        key=lambda row: (str(row["combo"]), not base.as_bool(row["sig_fdr"]), float(row["p"])),
    )
    assert len(conditional) == 5 and all(row["window"] == "5yr" for row in conditional)
    add_forest_pages(
        manifest,
        labels=["9C3"],
        filenames=["figure9c3_utilwas_ed_conditional_rate_5yr_forest.png"],
        page_rows=[conditional],
        kicker_domain="Emergency-department UtilWAS",
        title="Five-year ED encounter rates among patients with the diagnosis present",
        subtitle="The sole FDR-significant conditional-rate result plus two lowest-P FDR-null comparators per ED diagnosis role.",
        effect_type="IRR",
        context=lambda row: f"5-year · {'primary' if row['combo'] == 'ED_primary' else 'secondary'} ED|{base.forest_group_label(row['detail'])}",
        band_key=lambda row: str(row["combo"]),
        domain="UtilWAS, ED visits only",
        selection_rule="Sole five-year FDR-significant conditional-rate row plus two lowest-P five-year FDR-null comparators per ED role; 5 rows",
        caption_notes="Conditional IRRs apply only among patients with at least one qualifying ED encounter for the listed diagnosis; both windows remain in the supplemental atlas.",
        footer="Conditional-rate estimates are not diagnosis-presence estimates.",
    )

    q_binary = sorted(
        [row for row in qwas if row["form"] == "binary" and base.as_bool(row["sig_fdr"])],
        key=lambda row: (str(row["group"]).lower(), str(row["label"]).lower()),
    )
    q_continuous = sorted(
        [row for row in qwas if row["form"] == "continuous" and base.as_bool(row["sig_fdr"])],
        key=lambda row: (str(row["group"]).lower(), str(row["label"]).lower()),
    )
    assert len(q_binary) == 44 and len(q_continuous) == 37
    add_forest_pages(
        manifest,
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
        caption_notes="Asterisks identify referral-driving ascertainment-sensitive symptoms; null is 1.",
        footer="* Referral-driving symptom; large estimates may partly reflect the sleep-study referral pathway.",
    )
    add_forest_pages(
        manifest,
        labels=["10A2a", "10A2b"],
        filenames=["figure10a2a_qwas_continuous_forest.png", "figure10a2b_qwas_continuous_forest.png"],
        page_rows=[q_continuous[:19], q_continuous[19:]],
        kicker_domain="Questionnaire-wide association study",
        title="All FDR-significant continuous questionnaire associations",
        subtitle="β values are adjusted differences in rank-inverse-normal scores (approximately SD units).",
        effect_type="β",
        context=lambda row: str(row["group"]),
        band_key=lambda row: str(row["group"]),
        domain="QWAS",
        selection_rule="All 37 FDR-significant continuous QWAS rows, split 19/18",
        caption_notes="Higher FOSQ scores indicate better function; asterisks mark referral-driving symptoms.",
        footer="* Referral-driving symptom; higher FOSQ scores indicate better function.",
    )

    beh_binary = sorted(
        [row for row in beh if row["effect_type"] == "OR"],
        key=lambda row: ({"5yr": 0, "lifetime": 1}[str(row["window"])], str(row["label"]).lower()),
    )
    beh_continuous = sorted(
        [row for row in beh if row["effect_type"] == "beta"],
        key=lambda row: ({"5yr": 0, "lifetime": 1}[str(row["window"])], str(row["label"]).lower()),
    )
    assert len(beh_binary) == 7 and len(beh_continuous) == 5
    assert len({str(row["outcome"]) for row in beh_binary + beh_continuous}) == 12
    add_forest_pages(
        manifest,
        labels=["10B1"],
        filenames=["figure10b1_behwas_binary_forest.png"],
        page_rows=[beh_binary],
        kicker_domain="Behavior-wide association study",
        title="All unique binary behavior items",
        subtitle="Five-year estimates are used for windowed outcomes; undated outcomes retain their lifetime definition.",
        effect_type="OR",
        context=lambda row: "5-year window" if row["window"] == "5yr" else "Undated / lifetime",
        band_key=lambda row: str(row["window"]),
        domain="BehWAS",
        selection_rule="All 7 unique binary behavior items: five-year result for windowed outcomes and lifetime result for undated outcomes",
        caption_notes="Medication adherence is among users; FDR-null rows are retained for complete item coverage; one-year duplicates are omitted.",
        footer="Medication adherence is evaluated among medication users. Undated/lifetime outcomes are descriptive; BehWAS is preliminary.",
    )
    add_forest_pages(
        manifest,
        labels=["10B2"],
        filenames=["figure10b2_behwas_continuous_forest.png"],
        page_rows=[beh_continuous],
        kicker_domain="Behavior-wide association study",
        title="All unique continuous behavior items",
        subtitle="Five-year estimates are used for windowed outcomes; undated outcomes retain their lifetime definition.",
        effect_type="β",
        context=lambda row: "5-year window" if row["window"] == "5yr" else "Undated / lifetime",
        band_key=lambda row: str(row["window"]),
        domain="BehWAS",
        selection_rule="All 5 unique continuous behavior items: five-year result for windowed outcomes and lifetime result for undated outcomes",
        caption_notes="Medication PDC is among users; FDR-null rows are retained for complete item coverage; one-year duplicates are omitted.",
        footer="Medication PDC is evaluated among medication users. Undated/lifetime outcomes are descriptive; BehWAS is preliminary.",
    )
    return manifest


def heat_color(value: float) -> str:
    clipped = max(-0.85, min(0.85, value)) / 0.85
    endpoint = ImageColor.getrgb(TEAL if clipped >= 0 else PURPLE)
    white = ImageColor.getrgb("#FFFFFF")
    strength = abs(clipped) ** 0.78
    rgb = tuple(round(white[index] * (1 - strength) + endpoint[index] * strength) for index in range(3))
    return "#" + "".join(f"{channel:02X}" for channel in rgb)


def build_figure11() -> dict[str, object]:
    source_rows = [
        row
        for row in read_csv("results/incwas_results/phenotype_exposure/website_exports/octant_cluster_distributions.csv")
        if row["octant"] != "overall"
    ]
    assert all(not base.as_bool(row["suppressed"]) for row in source_rows)
    octants = [
        ("mild-all", "Mild All"),
        ("hypoxemia-predominant", "Hypoxemia Predominant"),
        ("symptom-predominant", "Symptom Predominant"),
        ("comorbidity-predominant", "Comorbidity Predominant"),
        ("hypoxemic-symptomatic", "Hypoxemic Symptomatic"),
        ("hypoxemic-comorbid", "Hypoxemic Comorbid"),
        ("symptomatic-comorbid", "Symptomatic Comorbid"),
        ("high-all", "High All"),
    ]
    metric_groups = [
        ("PHYSIOLOGY", ["odi4pct", "min_spo2", "mean_spo2", "t90_fraction"]),
        ("SYMPTOMS", ["ess_total", "isi_total", "fosq_impair", "phq2_total", "stop_total"]),
        (
            "COMORBIDITY BURDEN",
            [
                "n_phecodes_distinct",
                "cond_obesity",
                "cond_hyperlipidemia",
                "cond_hypertension",
                "cond_impaired_fasting_glucose",
                "cond_anxiety",
                "cond_gerd",
            ],
        ),
        (
            "AHI, DEMOGRAPHICS, AND EXTERNAL CHARACTERISTICS",
            [
                "index_ahi",
                "age_years",
                "bmi",
                "female",
                "ahi_severity_none",
                "ahi_severity_mild",
                "ahi_severity_moderate",
                "ahi_severity_severe",
            ],
        ),
        ("FOLLOW-UP AND UTILIZATION", ["followup_years", "enc_per_year"]),
    ]
    metric_order = [metric for _, metrics in metric_groups for metric in metrics]
    assert len(metric_order) == 26 and len(source_rows) == 26 * 8
    by_key = {(row["metric_id"], row["octant"]): row for row in source_rows}
    assert len(by_key) == 26 * 8
    metric_labels = {
        "odi4pct": "ODI 4% (events/h)",
        "min_spo2": "Minimum SpO₂ (%)",
        "mean_spo2": "Mean SpO₂ (%)",
        "t90_fraction": "T90 (% of sleep)",
        "ess_total": "Epworth total (points)",
        "isi_total": "Insomnia Severity Index (points)",
        "fosq_impair": "FOSQ impairment (points)",
        "phq2_total": "PHQ-2 (points)",
        "stop_total": "STOP score (points)",
        "n_phecodes_distinct": "Distinct Phecodes (count)",
        "cond_obesity": "Obesity",
        "cond_hyperlipidemia": "Hyperlipidemia",
        "cond_hypertension": "Hypertension",
        "cond_impaired_fasting_glucose": "Impaired fasting glucose",
        "cond_anxiety": "Anxiety disorders",
        "cond_gerd": "GERD",
        "index_ahi": "Index AHI (events/h)",
        "age_years": "Age (years)",
        "bmi": "BMI (kg/m²)",
        "female": "Female",
        "ahi_severity_none": "AHI <5 (no OSA)",
        "ahi_severity_mild": "AHI 5–<15 (mild)",
        "ahi_severity_moderate": "AHI 15–<30 (moderate)",
        "ahi_severity_severe": "AHI ≥30 (severe)",
        "followup_years": "IncWAS follow-up (years)",
        "enc_per_year": "Baseline encounters/year",
    }

    def cell_value(row: dict[str, str]) -> str:
        metric = row["metric_id"]
        value = float(row["estimate"])
        if row["metric_type"] == "binary":
            return f"{100 * value:.1f}%"
        if metric == "t90_fraction":
            return f"{100 * value:.1f}%"
        if metric in {"mean_spo2", "min_spo2"}:
            return f"{value:.1f}%"
        return f"{value:.1f}"

    height = 2500
    image = Image.new("RGB", (WIDTH, height), BG)
    draw = ImageDraw.Draw(image)
    top = draw_header(
        image,
        "Figure 11 · Phase 3 cross-domain phenotypes",
        "Clinical signature of the eight cross-domain phenotypes",
        "Cell values are aggregate means or percentages in native units; color encodes standardized difference from the shared cohort.",
        legend=False,
    )
    left = 56
    label_width = 370
    table_right = WIDTH - 56
    data_left = left + label_width
    column_width = (table_right - data_left) / len(octants)
    header_top = top + 6
    header_bottom = header_top + 154
    draw.rectangle((left, header_top, data_left, header_bottom), fill="#E5F0EE", outline="#B7CAC6", width=1)
    draw.text((left + 16, header_top + 54), "FEATURE", fill=TEAL, font=base.font(21, bold=True))
    cohort_ns: dict[str, int] = {}
    for column, (octant, label) in enumerate(octants):
        x0 = round(data_left + column * column_width)
        x1 = round(data_left + (column + 1) * column_width)
        draw.rectangle((x0, header_top, x1, header_bottom), fill="#E5F0EE", outline="#B7CAC6", width=1)
        cohort_n = int(by_key[("odi4pct", octant)]["n_nonmissing"])
        cohort_ns[octant] = cohort_n
        lines = base.wrap_to_width(draw, label, base.font(18, bold=True), x1 - x0 - 12)
        y = header_top + 20
        for line in lines[:3]:
            box = draw.textbbox((0, 0), line, font=base.font(18, bold=True))
            draw.text(((x0 + x1 - (box[2] - box[0])) / 2, y), line, fill=INK, font=base.font(18, bold=True))
            y += 21
        n_text = f"N={cohort_n:,}"
        n_box = draw.textbbox((0, 0), n_text, font=base.font(16))
        draw.text(((x0 + x1 - (n_box[2] - n_box[0])) / 2, header_bottom - 29), n_text, fill=MID_INK, font=base.font(16))
    assert sum(cohort_ns.values()) == 70880

    y = header_bottom
    group_height = 30
    row_height = 67
    row_index = 0
    for group_name, metrics in metric_groups:
        draw.rectangle((left, y, table_right, y + group_height), fill="#D8E9E6")
        draw.text((left + 12, y + 6), group_name, fill=TEAL, font=base.font(17, bold=True))
        y += group_height
        for metric in metrics:
            row_fill = "#FBFCFB" if row_index % 2 == 0 else "#F3F7F6"
            draw.rectangle((left, y, data_left, y + row_height), fill=row_fill, outline="#D6E1DF", width=1)
            label_lines = base.wrap_to_width(draw, metric_labels[metric], base.font(17, bold=True), label_width - 24)[:2]
            label_y = y + (row_height - len(label_lines) * 20) / 2
            for line_no, line in enumerate(label_lines):
                draw.text((left + 12, label_y + line_no * 20), line, fill=INK, font=base.font(17, bold=True))
            for column, (octant, _) in enumerate(octants):
                row = by_key[(metric, octant)]
                standardized = float(row["standardized_difference"])
                x0 = round(data_left + column * column_width)
                x1 = round(data_left + (column + 1) * column_width)
                fill = heat_color(standardized)
                draw.rectangle((x0, y, x1, y + row_height), fill=fill, outline="#D6E1DF", width=1)
                value = cell_value(row)
                value_font = base.font(18, bold=True)
                box = draw.textbbox((0, 0), value, font=value_font)
                text_fill = WHITE if abs(standardized) >= 0.58 else INK
                draw.text(((x0 + x1 - (box[2] - box[0])) / 2, y + 13), value, fill=text_fill, font=value_font)
                n_text = f"n={int(row['n_nonmissing']):,}"
                n_font = base.font(13)
                n_box = draw.textbbox((0, 0), n_text, font=n_font)
                draw.text(((x0 + x1 - (n_box[2] - n_box[0])) / 2, y + 40), n_text, fill=text_fill, font=n_font)
            y += row_height
            row_index += 1
    assert y <= 2340, y
    legend_y = y + 24
    draw.text((left, legend_y), "STANDARDIZED DIFFERENCE", fill=TEAL, font=base.font(17, bold=True))
    gradient_left = left + 260
    gradient_width = 510
    for pixel in range(gradient_width):
        value = -0.85 + 1.70 * pixel / max(gradient_width - 1, 1)
        draw.line((gradient_left + pixel, legend_y, gradient_left + pixel, legend_y + 22), fill=heat_color(value), width=1)
    draw.rectangle((gradient_left, legend_y, gradient_left + gradient_width, legend_y + 22), outline="#93A7A4", width=1)
    draw.text((gradient_left - 6, legend_y + 29), "lower", fill=MID_INK, font=base.font(15))
    mid_text = "cohort-like"
    mid_box = draw.textbbox((0, 0), mid_text, font=base.font(15))
    draw.text((gradient_left + gradient_width / 2 - (mid_box[2] - mid_box[0]) / 2, legend_y + 29), mid_text, fill=MID_INK, font=base.font(15))
    draw.text((gradient_left + gradient_width - 42, legend_y + 29), "higher", fill=MID_INK, font=base.font(15))
    n_values = [int(row["n_nonmissing"]) for row in source_rows]
    note = (
        f"Each cell shows its feature-specific denominator. Denominators range from {min(n_values):,} to {max(n_values):,}; "
        "missingness varies by feature and phenotype. T90 is displayed as percent of sleep. No race breakdown is shown."
    )
    for line_no, line in enumerate(base.wrap_to_width(draw, note, base.font(17, italic=True), WIDTH - 930)):
        draw.text((900, legend_y + line_no * 21), line, fill=MID_INK, font=base.font(17, italic=True))
    filename = "figure11_octant_clinical_signature.png"
    path = base.save_page(image, filename)
    denominators = {
        metric: {octant: int(by_key[(metric, octant)]["n_nonmissing"]) for octant, _ in octants}
        for metric in metric_order
    }
    coverage = {
        metric: {octant: float(by_key[(metric, octant)]["coverage_pct"]) for octant, _ in octants}
        for metric in metric_order
    }
    return {
        "figure": "11",
        "file": path.name,
        "width": WIDTH,
        "height": height,
        "rows": 26,
        "columns": 8,
        "domain": "Phase 3 cross-domain phenotypes",
        "effect_type": "native-unit aggregate summary with standardized-difference color",
        "selection_rule": "All 26 release-audited metrics across all eight octant phenotypes in the shared cohort (N=70,880); no race breakdown",
        "caption_notes": f"Cell annotations are aggregate means or percentages in native units and include feature-specific n. Color is standardized difference from the shared cohort, clipped at ±0.85. Feature-specific denominators range from {min(n_values):,} to {max(n_values):,}; exact denominators and coverage percentages are recorded in this manifest. T90 is displayed as percent of sleep.",
        "metric_groups": {group: metrics for group, metrics in metric_groups},
        "metric_labels": metric_labels,
        "phenotype_n": cohort_ns,
        "feature_denominators": denominators,
        "feature_coverage_pct": coverage,
    }


def step_curve(
    draw: ImageDraw.ImageDraw,
    times: list[float],
    values: list[float],
    box: tuple[int, int, int, int],
    y_max: float,
    color: str,
    width: int,
) -> None:
    x0, y0, x1, y1 = box
    points: list[tuple[float, float]] = []
    for index, (time, value) in enumerate(zip(times, values)):
        x = x0 + min(max(time, 0.0), 3.0) / 3.0 * (x1 - x0)
        y = y1 - min(max(value, 0.0), y_max) / y_max * (y1 - y0)
        if index and points:
            points.append((x, points[-1][1]))
        points.append((x, y))
    if len(points) >= 2:
        draw.line(points, fill=color, width=width, joint="curve")


def build_figure12() -> dict[str, object]:
    survival = read_json("website/public/data/phenotype-survival/phecode/428-2.json")
    metadata = read_csv("results/incwas_results/phenotype_exposure/website_exports/octant_cif_metadata.csv")
    distributions = read_csv("results/incwas_results/phenotype_exposure/website_exports/octant_cluster_distributions.csv")
    selected = [
        ("high-all", "High All"),
        ("hypoxemic-comorbid", "Hypoxemic Comorbid"),
        ("hypoxemic-symptomatic", "Hypoxemic Symptomatic"),
    ]
    metadata_by_octant = {
        row["octant"]: row
        for row in metadata
        if row["level"] == "phecode" and row["outcome"] == "428.2" and row["octant"] in {item[0] for item in selected}
    }
    ahi_by_octant = {
        row["octant"]: row
        for row in distributions
        if row["metric_id"] == "index_ahi" and row["octant"] in {item[0] for item in selected}
    }
    panels_by_octant = {
        str(panel["octant"]): panel
        for panel in survival["panels"]  # type: ignore[index]
        if str(panel["octant"]) in {item[0] for item in selected}
    }
    assert set(metadata_by_octant) == set(ahi_by_octant) == set(panels_by_octant) == {item[0] for item in selected}
    assert all(panel["curve_status"] == "available" and panel["curves"] for panel in panels_by_octant.values())

    height = 2500
    image = Image.new("RGB", (WIDTH, height), BG)
    draw = ImageDraw.Draw(image)
    top = draw_header(
        image,
        "Figure 12 · Phase 3 cross-domain phenotypes",
        "Heart failure incidence by selected cross-domain phenotype",
        "Aalen–Johansen cumulative incidence for heart failure NOS (Phecode 428.2); focal phenotype versus its pooled other-seven comparator.",
        legend=False,
    )
    boxes = base.grid_boxes(top + 10, 2360, 1, 3, gap=22)
    y_max = 4.5
    panel_letters = ["A", "B", "C"]
    for panel_letter, (octant, label), outer_box in zip(panel_letters, selected, boxes):
        ox0, oy0, ox1, oy1 = outer_box
        draw.rounded_rectangle(outer_box, radius=12, fill=PANEL_BG, outline="#C7D5D2", width=2)
        draw.text((ox0 + 20, oy0 + 14), f"{panel_letter}  {label}", fill=INK, font=base.font(30, bold=True))
        plot = (ox0 + 92, oy0 + 90, ox0 + 1130, oy1 - 100)
        px0, py0, px1, py1 = plot
        for tick in (0.0, 1.0, 2.0, 3.0):
            x = px0 + tick / 3.0 * (px1 - px0)
            draw.line((x, py0, x, py1), fill=GRID, width=1)
            text = f"{tick:.0f}"
            box = draw.textbbox((0, 0), text, font=base.font(18))
            draw.text((x - (box[2] - box[0]) / 2, py1 + 8), text, fill=MID_INK, font=base.font(18))
        for tick in (0.0, 1.5, 3.0, 4.5):
            y = py1 - tick / y_max * (py1 - py0)
            draw.line((px0, y, px1, y), fill=GRID, width=1)
            text = f"{tick:.1f}"
            box = draw.textbbox((0, 0), text, font=base.font(18))
            draw.text((px0 - 12 - (box[2] - box[0]), y - 10), text, fill=MID_INK, font=base.font(18))
        draw.line((px0, py0, px0, py1), fill=INK, width=2)
        draw.line((px0, py1, px1, py1), fill=INK, width=2)
        panel = panels_by_octant[octant]
        curves = panel["curves"]
        focal = curves["focal"]
        rest = curves["other_seven"]
        step_curve(draw, focal["time_years"], focal["cif_pct"], plot, y_max, ORANGE, 5)
        step_curve(draw, rest["time_years"], rest["cif_pct"], plot, y_max, "#7B8B8A", 4)
        draw.line((px0 + 18, py0 + 24, px0 + 65, py0 + 24), fill=ORANGE, width=5)
        draw.text((px0 + 76, py0 + 11), label, fill=INK, font=base.font(18, bold=True))
        draw.line((px0 + 330, py0 + 24, px0 + 377, py0 + 24), fill="#7B8B8A", width=4)
        draw.text((px0 + 388, py0 + 11), "Pooled other seven", fill=MID_INK, font=base.font(18, bold=True))
        xlabel = "Years since index"
        x_box = draw.textbbox((0, 0), xlabel, font=base.font(19, bold=True))
        draw.text(((px0 + px1 - (x_box[2] - x_box[0])) / 2, py1 + 36), xlabel, fill=MID_INK, font=base.font(19, bold=True))
        base.draw_rotated_centered(image, "Cumulative incidence (%)", (ox0 + 29, (py0 + py1) // 2), base.font(18, bold=True), MID_INK, angle=90)

        card = (ox0 + 1160, oy0 + 84, ox1 - 18, oy1 - 24)
        draw.rounded_rectangle(card, radius=10, fill=BAND_A, outline="#C9D9D6", width=1)
        cx, cy = card[0] + 20, card[1] + 16
        ahi = ahi_by_octant[octant]
        meta = metadata_by_octant[octant]
        stats = [
            ("Median index AHI", f"{float(ahi['median']):.1f} events/h"),
            ("IQR", f"{float(ahi['q1']):.1f}–{float(ahi['q3']):.1f}"),
            ("Adjusted m4 HR", f"{float(meta['hr_m4']):.2f} ({float(meta['ci_low']):.2f}–{float(meta['ci_high']):.2f})"),
            ("FDR q", f"{float(meta['q_fdr']):.3f}"),
            ("3-year CIF", f"{float(meta['cif3_focal_pct']):.2f}% focal"),
            ("Pooled-rest CIF", f"{float(meta['cif3_rest_pct']):.2f}%"),
            ("Events", f"{meta['events_focal']} focal / {meta['events_rest']} rest"),
        ]
        for stat_label, value in stats:
            draw.text((cx, cy), stat_label.upper(), fill=TEAL, font=base.font(16, bold=True))
            cy += 21
            draw.text((cx, cy), value, fill=INK, font=base.font(23, bold=True))
            cy += 40
        draw.rounded_rectangle((cx, cy + 4, card[2] - 18, cy + 44), radius=7, fill="#FFF1E7", outline=ORANGE, width=2)
        draw.text((cx + 12, cy + 14), "EPV <10 — interpret cautiously", fill="#8A3D16", font=base.font(18, bold=True))
        cy += 64
        draw.text((cx, cy), "AT RISK (FOCAL)", fill=TEAL, font=base.font(16, bold=True))
        cy += 23
        risk = panel["risk_table"]["focal"]
        risk_text = "  ·  ".join(f"{int(item['time_years'])}y: {item['n_at_risk']:,}" for item in risk)
        for line in base.wrap_to_width(draw, risk_text, base.font(17), card[2] - cx - 18):
            draw.text((cx, cy), line, fill=MID_INK, font=base.font(17))
            cy += 21

    footer = (
        "Each panel uses a different pooled-rest comparator because the focal phenotype is excluded from its own reference group. "
        "Curves are descriptive; HRs are adjusted one-versus-rest m4 estimates. EPV <10 flags limited events per covariate."
    )
    for line_no, line in enumerate(base.wrap_to_width(draw, footer, base.font(20, italic=True), WIDTH - 144)):
        draw.text((72, 2390 + line_no * 24), line, fill=MID_INK, font=base.font(20, italic=True))
    filename = "figure12_heart_failure_selected_phenotypes.png"
    path = base.save_page(image, filename)
    return {
        "figure": "12",
        "file": path.name,
        "width": WIDTH,
        "height": height,
        "rows": 3,
        "domain": "Phase 3 phenotype-associated incidence",
        "effect_type": "Aalen–Johansen CIF and adjusted cause-specific HR",
        "selection_rule": "Heart failure NOS (Phecode 428.2) for High All, Hypoxemic Comorbid, and Hypoxemic Symptomatic using release-approved public aggregate curves",
        "caption_notes": "Each panel compares one focal phenotype with a different pooled other-seven reference. Annotates median AHI (IQR), adjusted m4 HR (95% CI), FDR q, 3-year CIFs, event counts, and EPV<10 caution. No direct pairwise phenotype contrast is claimed.",
        "panels": {
            label: {
                "octant": octant,
                "median_ahi": float(ahi_by_octant[octant]["median"]),
                "ahi_iqr": [float(ahi_by_octant[octant]["q1"]), float(ahi_by_octant[octant]["q3"])],
                "hr": float(metadata_by_octant[octant]["hr_m4"]),
                "ci": [float(metadata_by_octant[octant]["ci_low"]), float(metadata_by_octant[octant]["ci_high"])],
                "q_fdr": float(metadata_by_octant[octant]["q_fdr"]),
                "cif3_focal_pct": float(metadata_by_octant[octant]["cif3_focal_pct"]),
                "cif3_rest_pct": float(metadata_by_octant[octant]["cif3_rest_pct"]),
                "quality": metadata_by_octant[octant]["unstable"],
            }
            for octant, label in selected
        },
    }


def validate_manifest(manifest: list[dict[str, object]]) -> None:
    expected_order = [
        "1A", "1B", "1C1", "1C2", "2A", "2B", "2C1", "2C2",
        "5A", "5B", "6A", "6B", "6C", "7A", "7B", "8A", "8B", "8C",
        "9A1", "9A2", "9B1", "9B2", "9C1", "9C2a", "9C2b", "9C3",
        "10A1a", "10A1b", "10A2a", "10A2b", "10B1", "10B2", "11", "12",
    ]
    observed_order = [str(entry["figure"]) for entry in manifest]
    assert observed_order == expected_order, observed_order
    for entry in manifest:
        path = OUT / str(entry["file"])
        with Image.open(path) as rendered:
            assert rendered.size == (entry["width"], entry["height"])
            assert rendered.mode == "RGB"
            assert rendered.height / rendered.width <= 1.35
            dpi = rendered.info.get("dpi")
            assert dpi and min(dpi) >= 299.0, (path.name, dpi)
        for required in ("figure", "domain", "effect_type", "rows", "selection_rule", "caption_notes"):
            assert required in entry, f"Manifest field {required!r} missing for {entry['file']}"
        for count in entry.get("label_counts", {}).values():
            assert int(count) <= 20


def main() -> None:
    data = base.load_data()
    observed = {key: len(value) for key, value in data.items()}
    expected = {"med": 395, "proc": 92, "util": 540, "qwas": 132, "beh": 20}
    assert observed == expected, observed
    prevalence, incidence = load_disease_data()
    manifest = (
        build_disease_figures(prevalence, incidence)
        + build_figures_5_to_8(data)
        + build_figures_9_to_10(data)
        + [build_figure11(), build_figure12()]
    )
    validate_manifest(manifest)
    output = {
        "source_primary_rows": {**observed, "prevalence": len(prevalence), "incidence": len(incidence)},
        "dpi": 300,
        "word_width_inches": WIDTH / 300,
        "assets": manifest,
    }
    manifest_path = OUT / "figure_manifest_260906.json"
    manifest_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
