"""Build an image-linked research summary for discussion with a clinician."""
from __future__ import annotations

from html import escape
from io import BytesIO
from pathlib import Path
import uuid

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                                TableStyle, PageBreak, Image, KeepTogether)

from settings import SOURCES, STAGES


def _clean(value, limit=1500):
    return str(value or "").strip()[:limit].replace("\x00", "")


def _register_fonts():
    regular = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    bold = regular.with_name("DejaVuSans-Bold.ttf")
    if regular.is_file() and bold.is_file():
        if "SkinSans" not in pdfmetrics.getRegisteredFontNames():
            pdfmetrics.registerFont(TTFont("SkinSans", str(regular)))
            pdfmetrics.registerFont(TTFont("SkinSansBold", str(bold)))
            pdfmetrics.registerFontFamily("SkinSans", normal="SkinSans", bold="SkinSansBold", italic="SkinSans", boldItalic="SkinSansBold")
        return "SkinSans", "SkinSansBold"
    return "Helvetica", "Helvetica-Bold"


def create_doctor_report(record, context, settings, run_dir=None):
    from skin_review_ui import result_language, safe_email, safe_website
    if not record:
        raise ValueError("Analyse an image before creating a report.")
    font, bold = _register_fonts()
    navy = colors.HexColor("#183743")
    teal = colors.HexColor("#0b706b")
    gray = colors.HexColor("#526772")
    border = colors.HexColor("#d9e5e3")
    styles = {
        "title": ParagraphStyle("TitleSkin", fontName=bold, fontSize=22, leading=27, textColor=navy, spaceAfter=13),
        "h2": ParagraphStyle("HeadingSkin", fontName=bold, fontSize=13, leading=18, textColor=navy, spaceBefore=12, spaceAfter=8),
        "body": ParagraphStyle("BodySkin", fontName=font, fontSize=9, leading=14, textColor=gray, spaceAfter=7),
        "small": ParagraphStyle("SmallSkin", fontName=font, fontSize=7.8, leading=11.3, textColor=gray, spaceAfter=6, splitLongWords=True),
        "label": ParagraphStyle("LabelSkin", fontName=bold, fontSize=8.5, leading=12, textColor=navy, spaceAfter=5),
        "caption": ParagraphStyle("CaptionSkin", fontName=font, fontSize=7.7, leading=11, textColor=gray, spaceAfter=5),
    }

    def p(value, kind="body", trusted=False):
        value = str(value) if trusted else escape(str(value)).replace("\n", "<br/>")
        return Paragraph(value, styles[kind])

    def table(rows, widths=(155, 340)):
        contents = [[p(row[0], "label"), p(row[1], "small")] for row in rows]
        out = Table(contents, colWidths=widths, hAlign="LEFT")
        out.setStyle(TableStyle([
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#f0f6f5")),
            ("LINEBELOW", (0, 0), (-1, -1), .45, border),
            ("LEFTPADDING", (0, 0), (-1, -1), 9),
            ("RIGHTPADDING", (0, 0), (-1, -1), 9),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]))
        return out

    def picture(image, width=232, height=147):
        if image is None:
            return p("Not available. No substitute image was generated.", "caption")
        buf = BytesIO()
        image.save(buf, format="PNG")
        buf.seek(0)
        image_width, image_height = image.size
        scale = min(width / image_width, height / image_height)
        return Image(buf, width=image_width * scale, height=image_height * scale, hAlign="CENTER")

    cfg = record["configuration"]
    directory = Path(record["report_dir"])
    directory.mkdir(parents=True, exist_ok=True)
    output = directory / (record["case_id"] + "_Doctor_Report_" + uuid.uuid4().hex[:6] + ".pdf")
    doc = SimpleDocTemplate(str(output), pagesize=(595.28, 841.89), rightMargin=50,
                           leftMargin=50, topMargin=82, bottomMargin=51,
                           title=settings.project_name + " - research image review",
                           author=settings.team_name)

    def decorate(canvas, document):
        canvas.saveState()
        canvas.setFillColor(teal)
        canvas.roundRect(50, 788, 27, 27, 8, fill=1, stroke=0)
        canvas.setFillColor(colors.white)
        canvas.setFont(bold, 11)
        canvas.drawCentredString(63.5, 797, "SR")
        canvas.setFillColor(navy)
        canvas.setFont(bold, 11)
        canvas.drawString(86, 803, _clean(settings.project_name, 45))
        canvas.setFont(font, 7)
        canvas.setFillColor(gray)
        canvas.drawString(86, 791, "RESEARCH OUTPUT - NOT A MEDICAL DIAGNOSIS")
        canvas.setStrokeColor(border)
        canvas.line(50, 777, 545, 777)
        canvas.line(50, 40, 545, 40)
        canvas.setFont(font, 7)
        canvas.drawString(50, 27, record["case_id"] + " | " + record["analysed_at"])
        canvas.drawRightString(545, 27, "Page " + str(document.page))
        canvas.restoreState()

    tone, headline, meaning, advice = result_language(record)
    story = []
    if record.get("demo_notice"):
        story += [p(record["demo_notice"], "label"), Spacer(1, 5)]
    story += [p("Image review for your clinician", "title"),
              p("This document records a research model output and optional user-entered notes. It does not establish a diagnosis, a referral or a treatment plan."),
              p(headline, "h2"), p(meaning), p(advice), Spacer(1, 6)]
    rows = [
        ("Model class", record["label"] + " (research prediction)"),
        ("Melanoma model score", f"{record['score']:.4f} - not a calibrated clinical probability"),
        ("Saved decision cutoff", f"{float(cfg['threshold']):.4f}; score at or above this cutoff is flagged"),
        ("Crop status", "Padded lesion ROI" if record["roi_reason"] == "cropped" else "Full-image fallback: " + record["roi_reason"]),
        ("Reference", _clean(context.get("Reference"), 80) or "Not provided"),
        ("Lesion location", _clean(context.get("Lesion location"), 120) or "Not provided"),
        ("First noticed", _clean(context.get("First noticed"), 120) or "Not provided"),
    ]
    story += [table(rows), p("Location, timing and reference are supplied by the user and are not verified or used by the model. Additional notes appear later in this report.", "small"),
              p("Image and estimated lesion outline", "h2")]
    pair = Table([[picture(record["original"]), picture(record["mask_overlay"])],
                  [p("Original supplied image", "caption"), p("Predicted mask overlay; this is not a cancer boundary.", "caption")]],
                 colWidths=[247.5, 247.5])
    pair.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 0)]))
    story += [pair, PageBreak(), p("How the software processed this image", "title"),
              p("These panels document the actual image analysis. No ground-truth mask is available for a new upload, so patient-specific Dice and IoU are not reported.")]
    stage_cells = []
    for index, (key, title, caption) in enumerate(STAGES):
        stage_cells.append([p(str(index + 1) + ". " + title, "label"),
                            picture(record.get(key), width=224, height=108), Spacer(1, 5), p(caption, "caption")])
    stage_cells.append([p("6. Classification", "label"),
                        p("The full image and lesion ROI are reviewed together by the dual-view EfficientNetB0 classifier."),
                        p("Model class: " + record["label"]),
                        p(f"Score {record['score']:.4f}; cutoff {float(cfg['threshold']):.4f}.")])
    grid = Table([stage_cells[:2], stage_cells[2:4], stage_cells[4:]], colWidths=[247.5, 247.5])
    grid.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"),
                              ("BOX", (0, 0), (-1, -1), .4, border),
                              ("INNERGRID", (0, 0), (-1, -1), .4, border),
                              ("LEFTPADDING", (0, 0), (-1, -1), 9),
                              ("RIGHTPADDING", (0, 0), (-1, -1), 9),
                              ("TOPPADDING", (0, 0), (-1, -1), 9),
                              ("BOTTOMPADDING", (0, 0), (-1, -1), 8)]))
    story += [grid, PageBreak(), p("Context, limitations and follow-up", "title"),
              p("User-entered notes", "h2"),
              p(_clean(context.get("Changes / symptoms / notes")) or "No additional notes supplied."),
              p("These notes are not used by the image model. They may be incomplete and require clinical confirmation.", "small"),
              p("Technical record", "h2")]
    story += [table([
        ("Pipeline", "Attention U-Net segmentation; dual-view EfficientNetB0 classification"),
        ("Saved inference methods", "Segmentation: " + str(cfg["segmentation_method"]) + "; classification: " + str(cfg["method"])),
        ("Image processing", "256 x 256 segmentation; 224 x 224 full-image and ROI classifier inputs. ROI padding " + str(cfg.get("roi_padding", .15)) + "; original quality-95 JPEG ROI encoding retained."),
        ("Segmentation cutoff", f"{float(cfg['segmentation_threshold']):.4f}"),
        ("Run", record["run_name"]),
        ("Source image", f"{record['source_dimensions'][0]} x {record['source_dimensions'][1]} pixels; SHA-256: {record['source_sha256']}"),
    ])]
    story += [p("What this output cannot tell you", "h2"),
              p("This student research prototype has not been validated for clinical diagnosis. It can miss melanoma and flag non-cancerous lesions. Its score is not an individual cancer risk percentage. It does not establish melanoma stage, severity, growth rate or lesion size in millimetres. Ordinary phone photos are outside the evaluated dermoscopic input setting."),
              p("The mask estimates the lesion's extent, not cancer cells. Grad-CAM is an explanation of the predicted class on the ROI branch, using an unflipped image. Each map is normalised separately. Heatmap appearance does not establish diagnostic correctness."),
              p("A clinician should assess a new, changing, bleeding or otherwise concerning spot regardless of this output. Examination, dermoscopy and sometimes biopsy are needed for diagnosis.")]
    if record.get("gradcam") is None:
        story.append(p("Grad-CAM was unavailable. No replacement explanation image was generated.", "label"))

    story += [p("Saved project evaluation", "h2"),
              p("These are aggregate exploratory project results, not this image's probability or segmentation accuracy. The ISIC test set was viewed during development; it is not independent clinical validation.", "small")]
    found = False
    if run_dir:
        for filename, title in [("complete_classification_test_metrics.csv", "Classification"),
                                ("complete_segmentation_test_metrics.csv", "Segmentation")]:
            path = Path(run_dir) / "tables" / filename
            if path.is_file():
                frame = pd.read_csv(path)
                if frame.empty:
                    continue
                # Include the saved table verbatim as text, rather than guessing units or definitions.
                story.append(p(title + " - saved table", "label"))
                for _, row in frame.iterrows():
                    line = "; ".join(f"{col}: {row[col]}" for col in frame.columns)
                    story.append(p(line, "small"))
                found = True
    if not found:
        story.append(p("No completed evaluation tables were available in the selected run. This report does not invent test loss or performance figures.", "small"))
    story += [p("Clinical support and project contact", "h2"),
              p("Australia: contact your GP or skin clinic for a skin assessment. Healthdirect nurse advice: 1800 022 222 (24 hours). Cancer Council / CancerConnect information and support: 13 11 20. In a medical emergency call 000."),
              p(settings.team_name + ". Project/software questions only; not a clinical advice service.", "small")]
    email, website = safe_email(settings.team_email), safe_website(settings.project_website)
    story.append(p("Project email: " + email if email else "Project contact email has not been configured.", "small"))
    if website:
        story.append(p(website, "small"))
    story += [p("Patient information sources", "h2")]
    for title, url in SOURCES:
        story.append(p(escape(title) + ': <link href="' + escape(url, quote=True) + '" color="#0b706b">' + escape(url) + '</link>', "small", trusted=True))
    story.append(p("Source links checked 16 October 2026. The software does not transmit this report to a healthcare provider.", "small"))
    doc.build(story, onFirstPage=decorate, onLaterPages=decorate)
    return str(output)
