"""Patient-facing presentation for the existing, frozen research pipeline.

No model training, threshold tuning or input re-encoding takes place here.
"""
from __future__ import annotations

from datetime import datetime
from html import escape
from pathlib import Path
from urllib.parse import urlparse
from zoneinfo import ZoneInfo
import hashlib
import logging
import math
import re
import shutil
import tempfile
import uuid
import warnings

import gradio as gr
import pandas as pd
from PIL import Image, UnidentifiedImageError

from settings import AppSettings, SOURCES, STAGES

LOGGER = logging.getLogger(__name__)
CSS = Path(__file__).with_name("app.css").read_text(encoding="utf-8")


def text(value):
    return escape(str(value), quote=True)


def safe_website(value):
    value = str(value).strip()
    parsed = urlparse(value)
    return value if parsed.scheme == "https" and parsed.netloc else ""


def safe_email(value):
    value = str(value).strip()
    return value if re.fullmatch(r"[A-Za-z0-9.!#$%&'*+/=?^_`{|}~-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", value) else ""


ICON = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" aria-hidden="true"><path d="M12 3 4 6v6c0 4 4 7 8 9 4-2 8-5 8-9V6l-8-3Z"/><path d="m8 12 3 3 5-6"/></svg>'
DOCUMENT_ICON = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" aria-hidden="true"><path d="M6 3h8l4 4v14H6V3Z"/><path d="M14 3v5h4M9 12h6M9 16h6"/></svg>'


def validate_image(path):
    if not path:
        raise ValueError("Choose a dermoscopic JPG or PNG image first.")
    path = Path(path)
    if not path.is_file():
        raise ValueError("The image is no longer available. Please upload it again.")
    if path.stat().st_size > 20 * 1024 * 1024:
        raise ValueError("Choose an image smaller than 20 MB.")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(path) as image:
                if image.format not in {"JPEG", "PNG"}:
                    raise ValueError("Only JPG and PNG images are supported.")
                if min(image.size) < 64 or image.width * image.height > 25_000_000:
                    raise ValueError("Use an image at least 64 pixels on each side and no larger than 25 megapixels.")
                image.verify()
            with Image.open(path) as image:
                return image.convert("RGB")
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError,
            Image.DecompressionBombWarning) as exc:
        raise ValueError("This file could not be read as a supported image. Try a different JPG or PNG.") from exc


def empty_summary():
    return f'''<section class="result-card" aria-live="polite">
      <div class="status-icon">{DOCUMENT_ICON}</div>
      <div class="eyebrow">Your image review</div>
      <h2>Start with a dermoscopic image.</h2>
      <p class="result-copy">Upload an image and select <strong>Analyse image</strong>. Your result will appear here with a clear explanation and practical next steps.</p>
      <div class="result-callout">Concerned about a new or changing spot? You can book a skin check with a GP without using this tool.</div>
    </section>'''


def result_language(record):
    if record["roi_reason"] != "cropped":
        return (
            "limited", "Limited image analysis", "The software could not reliably create a lesion crop.",
            "The classifier used a resized full image for its ROI input. This result needs particular caution. A clinician should assess any concerning spot.",
        )
    if record["predicted_label"] == 1:
        return (
            "flagged", "Review this spot with a clinician.", "The research model flagged this image for melanoma review.",
            "This does not mean you have melanoma. Book a skin check with a GP or skin clinic; a clinician can examine the spot and decide whether further tests are needed.",
        )
    return (
        "not-flagged", "The model did not flag this image.", "The image fell below this research model's melanoma cutoff.",
        "This does not rule out melanoma or another skin cancer. A new, changing, bleeding or unusual spot still needs assessment by a clinician.",
    )


def render_summary(record):
    tone, headline, meaning, advice = result_language(record)
    badge = "IMAGE ANALYSIS LIMITATION" if tone == "limited" else "YOUR IMAGE REVIEW"
    cam_note = " Heatmap unavailable; see the image process section." if record.get("gradcam") is None else ""
    return f'''<section class="result-card {tone}" aria-live="polite">
      <div class="status-icon">{ICON}</div><div class="eyebrow">{badge}</div>
      <h2>{text(headline)}</h2><p class="result-copy">{text(meaning)}</p>
      <div class="result-callout">{text(advice)}</div>
      <p class="case-meta">Review {text(record['case_id'])} &nbsp;·&nbsp; {text(record['analysed_at'])}{text(cam_note)}</p>
    </section>'''


def render_technical(record):
    if not record:
        return '<div class="metrics">The saved model output and fixed cutoff appear here after analysis.</div>'
    cfg = record["configuration"]
    fallback = ("Padded lesion crop" if record["roi_reason"] == "cropped" else
                "Full-image fallback: " + record["roi_reason"])
    explanation = ("Grad-CAM is calculated for the predicted class on the ROI branch. "
                   "It uses the unflipped image, including when classification averages four flips. "
                   "Each heatmap is normalised separately; brighter colours do not measure cancer severity.")
    if record.get("gradcam") is None:
        explanation = "Grad-CAM could not be generated for this image. No replacement heatmap has been invented."
    return f'''<div class="metrics">
      <strong>Technical output for a clinician or project reviewer</strong>
      <div class="metric-pair"><div class="metric">Model class<b>{text(record['label'])}</b></div>
      <div class="metric">Melanoma model score<b>{record['score']:.4f}</b></div>
      <div class="metric">Fixed decision cutoff<b>{float(cfg['threshold']):.4f}</b></div></div>
      <p>The model flags melanoma when its score is at or above the saved cutoff. The score is <strong>not a calibrated probability</strong> of cancer and cannot be used to assess severity or stage.</p>
      <p>Crop: {text(fallback)}. Segmentation cutoff: {float(cfg['segmentation_threshold']):.4f}. Classification: {text(cfg['method'])}. Segmentation: {text(cfg['segmentation_method'])}.</p>
      <p>{text(explanation)}</p>
      <p>There is no reference mask for a new upload, so a patient-specific Dice or IoU score cannot be calculated here.</p>
    </div>'''


NEXT_STEPS = '''<section class="next-steps">
 <div class="section-kicker">A practical way forward</div><h2>What to do next</h2>
 <div class="step-grid">
 <div class="step-item"><span class="step-number">1</span><div><h3>Arrange a skin check</h3><p>Speak with a GP or skin clinic about a concerning spot. A new or changing spot needs medical review, whatever the model says. <a href="https://www.healthdirect.gov.au/australian-health-services" target="_blank" rel="noopener noreferrer">Find a health service</a>.</p></div></div>
 <div class="step-item"><span class="step-number">2</span><div><h3>Make a note of changes</h3><p>Tell the clinician where the spot is, when you noticed it and whether it has changed, bled or become uncomfortable. Do not delay a skin check while waiting for this software.</p></div></div>
 <div class="step-item"><span class="step-number">3</span><div><h3>Take the report with you</h3><p>Add optional notes below, then download the doctor report. Share it through your usual healthcare channel. The software does not send it automatically.</p></div></div>
 </div></section>'''

UNDERSTANDING = '''<div class="info-panel"><div class="info-grid">
 <div class="info-box"><h3>If the image is flagged</h3><p>The model's score met its saved melanoma cutoff. This is a research prediction, not a diagnosis. Non-cancerous lesions can also be flagged.</p></div>
 <div class="info-box"><h3>If the image is not flagged</h3><p>The score was below the cutoff. Melanoma can be missed, so this result cannot confirm that a spot is harmless or replace an examination.</p></div>
 </div><h3>How a diagnosis is made</h3><p>A GP or skin specialist can examine the spot, often with a dermatoscope. A biopsy and laboratory examination may be needed to determine whether it is melanoma. The model cannot make that diagnosis.</p>
 <p><a href="https://connect.cancer.org.au/cancer-information/types-of-cancer/melanoma/diagnosis" target="_blank" rel="noopener noreferrer">Cancer Council: how melanoma is diagnosed</a></p></div>'''

EDUCATION = '''<div class="info-panel"><h3>What is melanoma?</h3>
 <p>Melanoma is a cancer that begins in the skin's pigment-producing cells. It can appear as a new spot or a change in an existing spot, and can occur on skin that is not regularly exposed to the sun.</p>
 <div class="info-grid"><div class="info-box"><h3>Changes to discuss with a clinician</h3><p>A spot that is new, growing, changing shape or colour, bleeding, or looks different from your other spots deserves a medical assessment. Melanoma does not always have a typical appearance.</p></div>
 <div class="info-box"><h3>Photos have limits</h3><p>An image does not show your medical history, changes over time or an examination of the rest of your skin. This prototype was developed using dermoscopic images; everyday phone photos have not been validated for it.</p></div></div>
 <p style="margin-top:14px"><a href="https://connect.cancer.org.au/cancer-information/types-of-cancer/melanoma" target="_blank" rel="noopener noreferrer">Learn about melanoma</a> &nbsp;·&nbsp;
 <a href="https://connect.cancer.org.au/cancer-information/types-of-cancer/melanoma/signs-and-risk-factors" target="_blank" rel="noopener noreferrer">Signs and risk factors</a></p>
 <p style="font-size:12px">Patient information sources: Cancer Council and Healthdirect Australia. Links checked 16 October 2026.</p></div>'''


def render_contacts(settings):
    email = safe_email(settings.team_email)
    website = safe_website(settings.project_website)
    links = (f'<a href="mailto:{text(email)}">{text(email)}</a><br>' if email else "")
    links += f'<a href="{text(website)}" target="_blank" rel="noopener noreferrer">Project website</a><br>' if website else ""
    if not links:
        links = "Project contact details have not been configured.<br>"
    return f'''<div class="contacts-grid">
     <div class="contact-card"><h3>Unsure what to do?</h3><a class="phone" href="tel:1800022222">1800 022 222</a><br>Healthdirect nurse advice, 24 hours a day in Australia.<br><a href="https://www.healthdirect.gov.au/contact-us" target="_blank" rel="noopener noreferrer">About this service</a></div>
     <div class="contact-card"><h3>Cancer information &amp; support</h3><a class="phone" href="tel:131120">13 11 20</a><br>Cancer Council / CancerConnect information and support.<br><a href="https://connect.cancer.org.au/cancer-information/types-of-cancer/melanoma" target="_blank" rel="noopener noreferrer">Melanoma information</a></div>
     <div class="contact-card"><h3>{text(settings.team_name)}</h3>{links}For project or software questions. This is not a clinical advice service.</div>
    </div><div class="info-panel" style="font-size:12px;padding-top:12px">These contacts are for Australia. In a medical emergency, call 000. A flagged image alone does not establish an emergency.</div>'''


class ReviewService:
    """Testable orchestration; the original model owns all inference decisions."""
    def __init__(self, pipeline, settings=None):
        self.pipeline = pipeline
        self.settings = settings or AppSettings()

    def analyse(self, path):
        validate_image(path)
        try:
            prediction = self.pipeline.predict_file(path, explain=True)
        except Exception:
            LOGGER.exception("Image analysis with Grad-CAM failed; retrying without the explanation.")
            prediction = self.pipeline.predict_file(path, explain=False)
        score = float(prediction["score"])
        cfg = dict(self.pipeline.configuration)
        if not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError("The model returned an invalid score. No result can be shown for this image.")
        if int(prediction["predicted_label"]) != int(score >= float(cfg["threshold"])):
            raise ValueError("The saved model result is inconsistent with its cutoff. Please check the model configuration.")
        record = dict(prediction)
        record.update({
            "case_id": "SR-" + uuid.uuid4().hex[:8].upper(),
            "analysed_at": datetime.now(ZoneInfo("Australia/Sydney")).strftime("%d %b %Y, %I:%M %p %Z"),
            "configuration": cfg,
            "source_sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
            "source_dimensions": list(prediction["original"].size),
            "run_name": self.pipeline.run_dir.name if self.pipeline.run_dir else "Run name unavailable",
            "demo_notice": self.settings.demo_notice,
            "report_dir": tempfile.mkdtemp(prefix="skin_review_report_"),
        })
        # The probability array is not required by the UI or PDF and can be large.
        record.pop("probability_map", None)
        return record

    def make_report(self, record, reference, location, first_seen, notes):
        if not record:
            raise ValueError("Analyse an image before creating its report.")
        from doctor_report import create_doctor_report
        context = {"Reference": reference, "Lesion location": location,
                   "First noticed": first_seen, "Changes / symptoms / notes": notes}
        return create_doctor_report(record, context, self.settings, self.pipeline.run_dir)


def discard_record(record):
    if record and record.get("report_dir"):
        shutil.rmtree(record["report_dir"], ignore_errors=True)


def build_demo(pipeline, settings=None, sample_paths=()):
    settings = settings or AppSettings()
    service = ReviewService(pipeline, settings)
    stage_images = []
    with gr.Blocks(title=settings.project_name + " | Image review", analytics_enabled=False,
                   delete_cache=(3600, 3600)) as demo:
        record = gr.State(None, time_to_live=3600, delete_callback=discard_record)
        gr.HTML(f'<header class="brandbar"><div class="brand"><span class="brandmark">{ICON}</span>{text(settings.project_name)}</div><span class="research-pill">Student research prototype</span></header>')
        gr.HTML('<div class="hero"><div class="eyebrow">Understand your image. Know your next step.</div><h1>A clearer view of your skin image.</h1><p>Explore a research model\'s image review, understand what it means and prepare information to discuss with a clinician.</p></div>')
        gr.HTML('<p class="scope-note"><strong>Not a medical diagnosis.</strong> Developed for dermoscopic research images; everyday phone photos have not been validated. A new or changing spot needs a clinical skin check.</p>')
        if settings.demo_notice:
            gr.HTML('<div class="demo-notice">' + text(settings.demo_notice) + '</div>')
        with gr.Row(elem_id="analysis-row"):
            with gr.Column(scale=4, min_width=300, elem_id="upload-panel"):
                gr.HTML('<h2 class="panel-heading">1. Add your image</h2><p class="panel-caption">Use a de-identified dermoscopic image. JPG or PNG, up to 20 MB.</p>')
                upload = gr.File(label="Choose or drop an image", file_types=[".jpg", ".jpeg", ".png"],
                                 type="filepath", elem_id="upload-file", height=115)
                preview = gr.Image(label="Image preview", interactive=False, height=225,
                                   buttons=["fullscreen"], format="png", elem_id="upload-preview")
                with gr.Row():
                    analyse_button = gr.Button("Analyse image", variant="primary", scale=3, elem_id="analyse-button")
                    clear_button = gr.Button("Clear", scale=1, elem_id="clear-button")
                upload_status = gr.HTML('<div class="upload-status">No image selected yet.</div>')
                if sample_paths:
                    gr.Examples(examples=[[str(p)] for p in sample_paths], inputs=upload,
                                label="Or try a public ISIC research image", cache_examples=False)
            with gr.Column(scale=6, min_width=320, elem_id="result-wrap"):
                summary = gr.HTML(empty_summary(), elem_id="result-card")
                gr.HTML('<div class="result-footnote">The model can miss melanoma and can flag non-cancerous lesions. Results support a project demonstration and cannot replace professional assessment.</div>')
        gr.HTML(NEXT_STEPS)
        with gr.Accordion("Understand your result", open=False, elem_classes="section-accordion"):
            gr.HTML(UNDERSTANDING)
            technical = gr.HTML(render_technical(None), elem_id="technical-output")
        with gr.Accordion("See your image, step by step", open=False, elem_classes="section-accordion", elem_id="pipeline-section"):
            gr.HTML('<p class="pipeline-intro">These images explain how the software reached its output. A lesion boundary and a heatmap answer different questions; neither establishes a diagnosis. Analyse an image to fill these panels.</p>')
            for row_index in range(2):
                with gr.Row():
                    for stage in STAGES[row_index * 3:(row_index + 1) * 3]:
                        key, title, caption = stage
                        with gr.Column(scale=1, min_width=220, elem_classes="stage-card"):
                            gr.HTML(f'<div class="stage-title">{STAGES.index(stage) + 1}. {text(title)}</div>')
                            stage_images.append(gr.Image(label=title, show_label=False, interactive=False,
                                                         height=205, buttons=["download", "fullscreen"],
                                                         format="png", elem_id="stage-" + key))
                            gr.HTML('<p class="stage-caption">' + text(caption) + '</p>')
                    if row_index == 1:
                        with gr.Column(scale=1, min_width=220, elem_classes="stage-card"):
                            gr.HTML('<div class="stage-title">6. The research prediction</div><div class="classification-stage">EfficientNetB0 reviews the full image and the lesion crop together.<br><br>Its score is compared with the saved decision cutoff to produce the model class.<br><br>Open <strong>Understand your result</strong> for the exact output.</div><p class="stage-caption">If a crop cannot be formed reliably, a full-image fallback is reported. No reference mask is available for a new upload.</p>')
        with gr.Accordion("Prepare a report for your doctor", open=False, elem_classes="section-accordion", elem_id="report-section"):
            gr.HTML('<div class="info-panel"><p>Download a PDF with the image, model output, pipeline images, limitations and your optional notes. It is a research summary for discussion, not a diagnosis or a referral.</p></div>')
            with gr.Row():
                reference = gr.Textbox(label="Case / patient reference (optional)", placeholder="Use a code rather than a full name", max_length=80)
                location = gr.Textbox(label="Where is the spot? (optional)", placeholder="For example, left forearm", max_length=120)
                first_seen = gr.Textbox(label="When did you first notice it? (optional)", placeholder="For example, about 2 months ago", max_length=120)
            notes = gr.Textbox(label="Changes, symptoms or questions for your doctor (optional)", lines=3,
                               placeholder="These notes are included in the report. They are not used by the model.", max_length=1500)
            report_button = gr.Button("Create doctor report", interactive=False, variant="primary", elem_id="create-report")
            report_file = gr.File(label="Download your doctor report (PDF)", interactive=False, elem_id="report-download")
            gr.HTML('<p class="report-note">Nothing is sent automatically. Download the file and share it with your clinician through your usual healthcare channel. Editing the notes clears the old download; create the report again to include your changes.</p>')
        with gr.Accordion("Learn about melanoma", open=False, elem_classes="section-accordion"):
            gr.HTML(EDUCATION)
        with gr.Accordion("Contact and support", open=False, elem_classes="section-accordion"):
            gr.HTML(render_contacts(settings))
        with gr.Accordion("Project evaluation and training curves", open=False, elem_classes="section-accordion"):
            gr.Markdown("### Saved project evaluation\nClassification and segmentation measure different tasks. These are exploratory project results: the ISIC test set was viewed during development. They are not independent clinical validation or this person's probability of cancer.")
            displayed = False
            if pipeline.run_dir:
                for filename, label in [("complete_segmentation_test_metrics.csv", "Lesion segmentation"),
                                        ("complete_classification_test_metrics.csv", "Melanoma classification")]:
                    path = pipeline.run_dir / "tables" / filename
                    if path.is_file():
                        table = pd.read_csv(path)
                        gr.Markdown("#### " + label)
                        gr.Dataframe(table, interactive=False, wrap=True)
                        displayed = True
                figures = sorted((pipeline.run_dir / "figures").glob("*.png"))
                if figures:
                    # Use decoded copies rather than exposing the model directory as a public path.
                    copies = []
                    for path in figures:
                        with Image.open(path) as handle:
                            copies.append((handle.convert("RGB"), path.stem.replace("_", " ")))
                    gr.Gallery(copies, columns=1, object_fit="contain", label="Saved evaluation figures")
                    displayed = True
            if not displayed:
                gr.Markdown("No saved evaluation tables or figures were found in this model run. Select the completed reporting run to include test loss, metrics and training curves.")
        gr.HTML('<div class="footer-note">Attention U-Net segmentation · Dual-view EfficientNetB0 classification.<br>Use de-identified research images. Uploads and reports may remain in runtime temporary storage until cleanup. This prototype is not a patient-record service. Do not upload identifiable patient material to a public demonstration.</div>')

        def reset(path, old_record):
            discard_record(old_record)
            image = None
            status = "No image selected yet."
            if path:
                try:
                    image = validate_image(path)
                    status = "Image ready. Select Analyse image."
                except ValueError as error:
                    status = str(error)
            return [image, None, empty_summary(), render_technical(None), *([None] * 5),
                    None, gr.Button(interactive=False), "", "", "", "",
                    '<div class="upload-status">' + text(status) + '</div>']

        def analyse(path, old_record):
            discard_record(old_record)
            try:
                result = service.analyse(path)
            except Exception as error:
                LOGGER.exception("Image analysis did not complete.")
                message = str(error) if isinstance(error, ValueError) else "Analysis could not be completed. Check the saved models and try again. No result has been produced."
                failure = f'<section class="result-card limited" aria-live="polite"><div class="eyebrow">Analysis unavailable</div><h2>We could not review this image.</h2><p class="result-copy">{text(message)}</p><div class="result-callout">If the spot concerns you, arrange a clinical skin check. You do not need this result to contact a GP.</div></section>'
                return [None, failure, render_technical(None), *([None] * 5), None, gr.Button(interactive=False)]
            return [result, render_summary(result), render_technical(result),
                    *[result.get(key) for key, _, _ in STAGES], None, gr.Button(interactive=True)]

        def report(result, ref, loc, seen, description):
            try:
                return service.make_report(result, ref, loc, seen, description)
            except Exception as error:
                LOGGER.exception("Report generation did not complete.")
                message = str(error) if isinstance(error, ValueError) else "The report could not be created. Please try again."
                raise gr.Error(message) from error

        case_outputs = [record, summary, technical, *stage_images, report_file, report_button]
        reset_outputs = [preview, record, summary, technical, *stage_images, report_file,
                         report_button, reference, location, first_seen, notes, upload_status]
        # Serialising case changes with inference prevents a previous image's result
        # from being attached to a newly selected image or report.
        upload.change(reset, [upload, record], reset_outputs, api_visibility="private",
                      concurrency_id="case_actions", concurrency_limit=1)
        analyse_button.click(analyse, [upload, record], case_outputs,
                             api_name="analyse_image", api_visibility="undocumented",
                             concurrency_id="case_actions", concurrency_limit=1)
        clear_button.click(lambda: None, outputs=upload, api_visibility="private")
        report_button.click(report, [record, reference, location, first_seen, notes], report_file,
                            api_name="create_doctor_report", api_visibility="undocumented",
                            concurrency_id="case_actions", concurrency_limit=1)
        for field in [reference, location, first_seen, notes]:
            field.change(lambda: None, outputs=report_file, api_visibility="private",
                         concurrency_id="case_actions", concurrency_limit=1)
    demo.skin_service = service
    demo.skin_callbacks = {"reset": reset, "analyse": analyse, "report": report}
    return demo


def launch_app(pipeline, settings=None, sample_paths=(), share=False, **kwargs):
    demo = build_demo(pipeline, settings, sample_paths)
    theme = gr.themes.Soft(primary_hue="teal", neutral_hue="slate", font=["Inter", "Arial", "sans-serif"])
    theme.set(body_background_fill="#f3f7f7", block_background_fill="#ffffff",
              button_primary_background_fill="#0b706b", button_primary_text_color="#ffffff",
              block_label_text_color="#34545c", block_label_background_fill="#edf4f2",
              block_title_text_color="#34545c", color_accent="#0b706b",
              link_text_color="#086b64", link_text_color_hover="#054f49")
    # Keep one coherent light palette even when the browser prefers dark mode.
    # Gradio's dark tokens otherwise mix dark inputs with the custom light layout.
    dark_values = {key: value for key in vars(theme) if key.endswith("_dark")
                   and isinstance((value := getattr(theme, key[:-5], None)), str)}
    theme.set(**dark_values)
    blocked = [str(pipeline.run_dir.resolve())] if pipeline.run_dir else []
    demo.queue(default_concurrency_limit=1).launch(theme=theme, css=CSS, share=share,
        max_file_size="20mb", blocked_paths=blocked, footer_links=[],
        show_error=False, **kwargs)
    return demo
