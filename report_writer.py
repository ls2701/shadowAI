"""
report_writer.py — turns rag_analyse.py's findings list directly into a PDF.

FIX vs the docx+LibreOffice version: that approach silently produced a
.docx with no PDF whenever LibreOffice wasn't installed on the machine
running backend_api.py (the common case on a bare Windows PC). This
version uses reportlab, a pure-Python PDF library with no external
binary dependency, so it always produces a real .pdf — nothing to be
missing, nothing to fall back from.

Each finding is a flat dict: classification, risk, confidence, application,
user, reason, evidence, policy, severity, recommended_action, plus the
_sensor/_source/_start_ts/_end_ts/_applications/_match_tier/_bytes/_event_count
fields backend_api.py adds. Any field may be missing or None if the LLM
call failed to parse — every access below uses .get() with a fallback.
"""

from datetime import datetime, timezone
from pathlib import Path
from xml.sax.saxutils import escape as _xml_escape

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, ListFlowable, ListItem, HRFlowable,
)

RISK_HEX = {
    "Critical": "#DC2626",
    "High": "#D97706",
    "Medium": "#2563EB",
    "Low": "#059669",
}
INK = "#1E293B"
MUTED = "#64748B"
BORDER = "#E2E8F0"


def _esc(val) -> str:
    """XML-escapes any text before it goes into a reportlab Paragraph.

    CRITICAL FIX: Paragraph() parses its input as a restricted XML/HTML-like
    markup language. Real log evidence and LLM-generated 'reason'/'evidence'
    text routinely contains a bare '<' with no matching '>' — an ARN
    placeholder like <account-id>, a redaction marker, a comparison chain
    like 'a<b<c', a truncated code snippet — and reportlab's parser throws
    a ValueError on any of these instead of just rendering the character.
    Without this escape, that ValueError propagates up through write_report()
    and crashes the whole pipeline job right after LLM analysis finishes and
    right before the report is written — with no PDF ever produced, and the
    frontend showing no download option because the job silently errored.
    Every value that originates from raw logs or LLM output goes through
    this before being interpolated into a Paragraph string. The font/color
    tags this file constructs itself are written directly, not through this
    function, so they remain real markup.
    """
    if val is None:
        return ""
    return _xml_escape(str(val))


styles = getSampleStyleSheet()
styles.add(ParagraphStyle("H1c", parent=styles["Heading1"], textColor=colors.HexColor(INK), spaceAfter=6))
styles.add(ParagraphStyle("H2c", parent=styles["Heading2"], textColor=colors.HexColor("#334155"),
                           spaceBefore=18, spaceAfter=8))
styles.add(ParagraphStyle("H3c", parent=styles["Heading3"], textColor=colors.HexColor("#334155"),
                           spaceBefore=14, spaceAfter=4))
styles.add(ParagraphStyle("Muted", parent=styles["Normal"], textColor=colors.HexColor(MUTED), fontSize=9))
styles.add(ParagraphStyle("Body", parent=styles["Normal"], fontSize=10, leading=14,
                           textColor=colors.HexColor(INK)))
styles.add(ParagraphStyle("Label", parent=styles["Normal"], fontSize=9, fontName="Helvetica-Bold",
                           textColor=colors.HexColor("#475569"), spaceBefore=6, spaceAfter=2))


def _severity(f: dict) -> str:
    for key in ("severity", "risk"):
        val = f.get(key)
        if val in RISK_HEX:
            return val
    return "Medium"


def _chip(text: str, hex_color: str):
    p = Paragraph(f'<font color="white"><b>{text}</b></font>',
                   ParagraphStyle("chip", parent=styles["Normal"], fontSize=8, alignment=1))
    t = Table([[p]], colWidths=[0.9 * inch], rowHeights=[0.22 * inch])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor(hex_color)),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
    ]))
    return t


def write_report(findings: list[dict], source_file: str, output_dir: str = "outputs") -> Path:
    Path(output_dir).mkdir(exist_ok=True, parents=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_path = Path(output_dir) / f"shadow_ai_report_{timestamp}.pdf"

    doc = SimpleDocTemplate(str(out_path), pagesize=letter,
                             topMargin=0.7 * inch, bottomMargin=0.7 * inch,
                             leftMargin=0.7 * inch, rightMargin=0.7 * inch)
    story = []

    counts = {"Critical": 0, "High": 0, "Medium": 0, "Low": 0}
    for f in findings:
        counts[_severity(f)] += 1

    # ── Cover ────────────────────────────────────────────────────
    story.append(Paragraph("Shadow AI Detection Report", styles["H1c"]))
    story.append(Paragraph(f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}", styles["Muted"]))
    story.append(Paragraph(f"Source file: {_esc(source_file)}", styles["Muted"]))
    story.append(Spacer(1, 16))

    def stat_cell(label, value, hex_color):
        p1 = Paragraph(f'<font size="18"><b>{value}</b></font>',
                        ParagraphStyle("v", parent=styles["Normal"], textColor=colors.HexColor(hex_color), alignment=1))
        p2 = Paragraph(label, ParagraphStyle("l", parent=styles["Muted"], alignment=1))
        t = Table([[p1], [p2]], colWidths=[1.6 * inch])
        t.setStyle(TableStyle([
            ("BOX", (0, 0), (-1, -1), 0.75, colors.HexColor(BORDER)),
            ("TOPPADDING", (0, 0), (-1, -1), 10), ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
        ]))
        return t

    stat_row = Table([[
        stat_cell("Total Findings", len(findings), INK),
        stat_cell("Critical", counts["Critical"], RISK_HEX["Critical"]),
        stat_cell("High", counts["High"], RISK_HEX["High"]),
        stat_cell("Medium", counts["Medium"], RISK_HEX["Medium"]),
        stat_cell("Low", counts["Low"], RISK_HEX["Low"]),
    ]], colWidths=[1.6 * inch] * 5)
    stat_row.setStyle(TableStyle([("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4)]))
    story.append(stat_row)
    story.append(Spacer(1, 20))

    # ── Executive summary ───────────────────────────────────────
    story.append(Paragraph("Executive Summary", styles["H2c"]))
    critical_high = [f for f in findings if _severity(f) in ("Critical", "High")]
    story.append(Paragraph(
        f"{len(findings)} AI-related sessions were detected in this log file. "
        f"{len(critical_high)} are rated High or Critical and require review.", styles["Body"]))
    story.append(Spacer(1, 8))
    items = []
    for f in critical_high[:8]:
        apps = ", ".join(f.get("_applications") or []) or f.get("application") or "Unknown"
        sev = _severity(f)
        reason_snip = (f.get("reason") or "")[:150]
        items.append(ListItem(Paragraph(
            f'<font color="{RISK_HEX[sev]}"><b>[{sev}]</b></font> {_esc(apps)} \u2014 {_esc(reason_snip)}',
            styles["Body"]), leftIndent=8))
    if items:
        story.append(ListFlowable(items, bulletType="bullet"))
    story.append(PageBreak())

    # ── Findings table ───────────────────────────────────────────
    story.append(Paragraph("All Findings", styles["H2c"]))
    header = ["User", "Application", "Severity", "Match", "Bytes", "Classification"]
    rows = [header]
    for f in findings:
        sev = _severity(f)
        apps = ", ".join(f.get("_applications") or []) or f.get("application") or "Unknown"
        rows.append([
            Paragraph(_esc(f.get("user") or "unknown"), styles["Body"]),
            Paragraph(_esc(apps), styles["Body"]),
            Paragraph(f'<font color="{RISK_HEX[sev]}"><b>{_esc(sev)}</b></font>', styles["Body"]),
            Paragraph(_esc(f.get("_match_tier", "")), styles["Body"]),
            Paragraph(_esc(f.get("_bytes", 0)), styles["Body"]),
            Paragraph(_esc(f.get("classification", "")), styles["Body"]),
        ])

    tbl = Table(rows, colWidths=[1.1 * inch, 1.1 * inch, 0.8 * inch, 0.8 * inch, 0.8 * inch, 1.3 * inch], repeatRows=1)
    style_cmds = [
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#F1F5F9")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor(INK)),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor(BORDER)),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]
    for i in range(1, len(rows)):
        if i % 2 == 0:
            style_cmds.append(("BACKGROUND", (0, i), (-1, i), colors.HexColor("#FAFAFA")))
    tbl.setStyle(TableStyle(style_cmds))
    story.append(tbl)
    story.append(PageBreak())

    # ── Detail sections ──────────────────────────────────────────
    story.append(Paragraph("Detailed Findings", styles["H2c"]))
    for idx, f in enumerate(findings, 1):
        sev = _severity(f)
        apps = ", ".join(f.get("_applications") or []) or f.get("application") or "Unknown"
        story.append(Paragraph(f"Finding {idx}: {_esc(apps)}", styles["H3c"]))

        meta_row = Table([[
            _chip(sev, RISK_HEX[sev]),
            Paragraph(f"{_esc(f.get('classification', 'Unknown'))} &nbsp;\u00b7&nbsp; "
                      f"Confidence: {_esc(f.get('confidence', 'n/a'))} &nbsp;\u00b7&nbsp; "
                      f"Match: {_esc(f.get('_match_tier', 'n/a'))}", styles["Body"]),
        ]], colWidths=[1 * inch, 5.3 * inch])
        meta_row.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "MIDDLE")]))
        story.append(meta_row)
        story.append(Spacer(1, 6))

        story.append(Paragraph("Reason", styles["Label"]))
        story.append(Paragraph(_esc(f.get("reason")) or "(not provided)", styles["Body"]))

        evidence = f.get("evidence")
        story.append(Paragraph("Evidence", styles["Label"]))
        ev_list = evidence if isinstance(evidence, list) and evidence else [str(evidence or "(none)")]
        story.append(ListFlowable(
            [ListItem(Paragraph(_esc(e), styles["Body"]), leftIndent=8) for e in ev_list],
            bulletType="bullet"))

        if f.get("policy"):
            story.append(Paragraph("Policy", styles["Label"]))
            story.append(Paragraph(_esc(f["policy"]), styles["Body"]))

        story.append(Paragraph("Recommended Action", styles["Label"]))
        story.append(Paragraph(
            f'<font color="{RISK_HEX[sev]}"><b>{_esc(f.get("recommended_action") or "(not provided)")}</b></font>',
            styles["Body"]))

        story.append(Paragraph("Session Info", styles["Label"]))
        story.append(Paragraph(
            f"Sensor/Source: {_esc(f.get('_sensor'))} / {_esc(f.get('_source'))} &nbsp;\u2022&nbsp; "
            f"Window: {_esc(f.get('_start_ts'))} \u2192 {_esc(f.get('_end_ts'))} &nbsp;\u2022&nbsp; "
            f"Events: {_esc(f.get('_event_count', 0))} &nbsp;\u2022&nbsp; Bytes: {_esc(f.get('_bytes', 0))}",
            styles["Muted"]))
        story.append(HRFlowable(width="100%", color=colors.HexColor(BORDER), spaceBefore=12, spaceAfter=12))

    doc.build(story)
    return out_path