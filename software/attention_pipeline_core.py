"""Inference and Gradio UI for the frozen Attention U-Net melanoma candidate.

No training or threshold selection occurs in this module. JPEG ROI encoding is
preserved because the training notebook classified saved quality-95 JPEG ROIs.
"""
from __future__ import annotations

from io import BytesIO
from pathlib import Path
import hashlib
import json
import threading

import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage
import tensorflow as tf

SEG_SIZE = (256, 256)
CLS_SIZE = (224, 224)


def image_tensor(path, size, normalise=False):
    pixels = tf.io.decode_image(tf.io.read_file(str(path)), channels=3,
                                expand_animations=False)
    pixels = tf.cast(tf.image.resize(pixels, size, method="bilinear"), tf.float32)
    return pixels / 255.0 if normalise else pixels


def roi_tensor(roi):
    # Match roi.save(..., quality=95) -> decode_classifier_image() in training.
    buffer = BytesIO()
    roi.convert("RGB").save(buffer, format="JPEG", quality=95)
    pixels = tf.io.decode_jpeg(buffer.getvalue(), channels=3)
    return tf.cast(tf.image.resize(pixels, CLS_SIZE), tf.float32)


def crop_from_prediction(image_pil, predicted_mask, padding=0.15):
    image = image_pil.convert("RGB")
    width, height = image.size
    binary = Image.fromarray(np.uint8(np.asarray(predicted_mask) > 0) * 255)
    binary = np.asarray(binary.resize(image.size, Image.Resampling.NEAREST)) > 0
    labelled, count = ndimage.label(binary)
    reason = "cropped"
    if count == 0:
        crop, reason = image, "empty predicted mask"
    else:
        sizes = np.bincount(labelled.ravel())
        sizes[0] = 0
        lesion = labelled == int(sizes.argmax())
        area = int(lesion.sum())
        if area < max(64, int(width * height * 0.001)) or area > width * height * 0.80:
            crop, reason = image, "implausible predicted area"
        else:
            ys, xs = np.where(lesion)
            span = max(np.ptp(xs) + 1, np.ptp(ys) + 1)
            margin = int(np.ceil(span * padding))
            x0 = max(0, int(xs.min()) - margin)
            x1 = min(width, int(xs.max()) + margin + 1)
            y0 = max(0, int(ys.min()) - margin)
            y1 = min(height, int(ys.max()) + margin + 1)
            crop = image.crop((x0, y0, x1, y1))
    a = np.asarray(crop)
    h, w = a.shape[:2]
    side = max(h, w)
    top, left = (side - h) // 2, (side - w) // 2
    square = np.pad(a, ((top, side - h - top), (left, side - w - left), (0, 0)),
                    mode="edge")
    roi = Image.fromarray(square).resize((CLS_SIZE[1], CLS_SIZE[0]),
                                         Image.Resampling.BILINEAR)
    return roi, reason


def mask_overlay(image, mask):
    base = np.asarray(image.convert("RGB"), dtype=np.float32)
    mask = Image.fromarray(np.uint8(mask) * 255).resize(
        image.size, Image.Resampling.NEAREST)
    binary = np.asarray(mask) > 127
    tint = np.array([35, 195, 145], dtype=np.float32)
    base[binary] = 0.65 * base[binary] + 0.35 * tint
    boundary = binary & ~ndimage.binary_erosion(binary)
    base[boundary] = [255, 215, 30]
    return Image.fromarray(np.uint8(np.clip(base, 0, 255)))


def gradcam_overlay(roi, heatmap):
    from matplotlib import colormaps
    image = roi.convert("RGB").resize((CLS_SIZE[1], CLS_SIZE[0]))
    heat = Image.fromarray(np.uint8(np.clip(heatmap, 0, 1) * 255)).resize(
        image.size, Image.Resampling.BILINEAR)
    colors = colormaps["jet"](np.asarray(heat, dtype=np.float32) / 255.0)[..., :3]
    base = np.asarray(image, dtype=np.float32) / 255.0
    return Image.fromarray(np.uint8(255 * np.clip(0.62 * base + 0.38 * colors, 0, 1)))


class MelanomaPipeline:
    def __init__(self, segmenter, classifier, configuration, run_dir=None):
        self.segmenter = segmenter
        self.classifier = classifier
        self.configuration = dict(configuration)
        self.run_dir = Path(run_dir) if run_dir else None
        self.lock = threading.RLock()
        cfg = self.configuration
        if cfg.get("view") != "dual":
            raise ValueError("This app requires the full-image + ROI classifier (view='dual').")
        for field in ("threshold", "segmentation_threshold"):
            if field not in cfg or not 0 <= float(cfg[field]) <= 1:
                raise ValueError(f"Missing or invalid saved configuration field: {field}")
        for field in ("method", "segmentation_method"):
            if cfg.get(field) not in ("single", "4-flip average"):
                raise ValueError(f"Unsupported inference method: {cfg.get(field)}")
        if segmenter.input_shape[1:] != (*SEG_SIZE, 3):
            raise ValueError("Segmenter input must be 256 x 256 RGB.")
        if len(classifier.inputs) != 2:
            raise ValueError("Expected two classifier inputs: full image and ROI.")
        shapes = [tuple(x.shape[1:]) for x in classifier.inputs]
        if shapes != [(*CLS_SIZE, 3), (*CLS_SIZE, 3)]:
            raise ValueError(f"Classifier input shapes are incompatible: {shapes}")

    @classmethod
    def load(cls, run_dir):
        run_dir = Path(run_dir)
        paths = {
            "configuration": run_dir / "tables" / "selected_configuration.json",
            "segmenter": run_dir / "models" / "attention_unet_best.keras",
            "classifier": run_dir / "models" / "selected_final_classifier.keras",
        }
        missing = [str(p) for p in paths.values() if not p.is_file()]
        if missing:
            raise FileNotFoundError("The saved pipeline is incomplete. Missing:\n" + "\n".join(missing))
        configuration = json.loads(paths["configuration"].read_text())
        provenance = run_dir / "tables" / "segmentation_configuration.json"
        if provenance.is_file():
            expected = json.loads(provenance.read_text()).get("checkpoint_sha256")
            if expected:
                actual = hashlib.sha256(paths["segmenter"].read_bytes()).hexdigest()
                if actual != expected:
                    raise ValueError("The segmentation checkpoint does not match its saved configuration.")
        manifest_path = run_dir / "deployment_manifest.json"
        if manifest_path.is_file():
            for relative, expected in json.loads(manifest_path.read_text()).get("sha256", {}).items():
                actual = hashlib.sha256((run_dir / relative).read_bytes()).hexdigest()
                if actual != expected:
                    raise ValueError(f"Deployment bundle file has changed: {relative}")
        # compile=False avoids needing custom training losses for inference.
        segmenter = tf.keras.models.load_model(paths["segmenter"], compile=False)
        classifier = tf.keras.models.load_model(paths["classifier"], compile=False)
        return cls(segmenter, classifier, configuration, run_dir)

    def segmentation_probabilities(self, images, batch_size=4):
        images = np.asarray(images, dtype=np.float32)
        methods = [(), (2,), (1,), (1, 2)] if self.configuration[
            "segmentation_method"] == "4-flip average" else [()]
        predictions = []
        for axes in methods:
            transformed = np.flip(images, axis=axes) if axes else images
            score = self.segmenter.predict(np.ascontiguousarray(transformed),
                                            batch_size=batch_size, verbose=0)[..., 0]
            score = np.flip(score, axis=axes) if axes else score
            predictions.append(score)
        return np.mean(np.stack(predictions), axis=0)

    def classifier_probabilities(self, full_images, roi_images, batch_size=8):
        full_images = np.asarray(full_images, dtype=np.float32)
        roi_images = np.asarray(roi_images, dtype=np.float32)
        methods = [(), (2,), (1,), (1, 2)] if self.configuration[
            "method"] == "4-flip average" else [()]
        predictions = []
        for axes in methods:
            full = np.flip(full_images, axis=axes) if axes else full_images
            roi = np.flip(roi_images, axis=axes) if axes else roi_images
            features = {"full_image": np.ascontiguousarray(full),
                        "roi_image": np.ascontiguousarray(roi)}
            predictions.append(self.classifier.predict(
                features, batch_size=batch_size, verbose=0).ravel())
        return np.mean(np.stack(predictions), axis=0)

    def gradcam(self, full_batch, roi_batch, target_class):
        model = self.classifier
        backbone = model.get_layer("efficientnetb0")
        gap = model.get_layer("gap")
        with tf.GradientTape() as tape:
            full_map = backbone(full_batch, training=False)
            roi_map = backbone(roi_batch, training=False)
            tape.watch(roi_map)
            features = model.get_layer("combine_features")([gap(full_map), gap(roi_map)])
            hidden = model.get_layer("hidden_dense")(features)
            hidden = model.get_layer("head_dropout")(hidden, training=False)
            output = model.get_layer("prediction")
            logit = tf.linalg.matmul(hidden, output.kernel) + output.bias
            score = logit[:, 0] if target_class == 1 else -logit[:, 0]
        gradients = tape.gradient(score, roi_map)
        if gradients is None:
            raise RuntimeError("Grad-CAM could not trace the ROI feature map.")
        weights = tf.reduce_mean(gradients, axis=(1, 2), keepdims=True)
        heatmap = tf.nn.relu(tf.reduce_sum(roi_map * weights, axis=-1)[0])
        return (heatmap / (tf.reduce_max(heatmap) + 1e-8)).numpy()

    def predict_file(self, path, explain=True):
        if path is None:
            raise ValueError("Upload a dermoscopic lesion image first.")
        path = Path(path)
        with Image.open(path) as handle:
            original = handle.convert("RGB")
        with self.lock:
            segmentation_input = image_tensor(path, SEG_SIZE, normalise=True)[None, ...]
            probabilities = self.segmentation_probabilities(segmentation_input)[0]
            mask = probabilities >= float(self.configuration["segmentation_threshold"])
            roi, reason = crop_from_prediction(
                original, mask, padding=float(self.configuration.get("roi_padding", 0.15)))
            full_input = image_tensor(path, CLS_SIZE)[None, ...]
            roi_input = roi_tensor(roi)[None, ...]
            score = float(self.classifier_probabilities(full_input, roi_input)[0])
            predicted = int(score >= float(self.configuration["threshold"]))
            heatmap = self.gradcam(full_input, roi_input, predicted) if explain else None
        result = {
            "original": original, "mask": Image.fromarray(np.uint8(mask) * 255),
            "mask_overlay": mask_overlay(original, mask), "roi": roi,
            "score": score, "predicted_label": predicted,
            "label": "Melanoma" if predicted else "Benign",
            "roi_reason": reason, "probability_map": probabilities,
        }
        result["gradcam"] = gradcam_overlay(roi, heatmap) if heatmap is not None else None
        return result


def build_demo(pipeline, sample_paths=()):
    import gradio as gr
    cfg = pipeline.configuration

    def preview(path):
        if path is None:
            return None
        with Image.open(path) as image:
            return image.convert("RGB")

    def analyse(path):
        try:
            result = pipeline.predict_file(path)
        except (ValueError, FileNotFoundError, OSError) as error:
            raise gr.Error(str(error)) from error
        summary = f"### Model prediction: **{result['label']}**\n"
        summary += (f"Melanoma model score: **{result['score']:.4f}**. "
                    f"Decision threshold: {float(cfg['threshold']):.4f}.\n\n"
                    "The score is a model output, not a calibrated clinical probability.")
        if result["roi_reason"] != "cropped":
            summary += "\n\nThe lesion crop was unreliable; the classifier used the full image for the ROI view."
        panels = [(result["original"], "Original image"),
                  (result["mask"], "Predicted lesion mask"),
                  (result["mask_overlay"], "Lesion outline"),
                  (result["roi"], "Lesion ROI"),
                  (result["gradcam"], "Grad-CAM on ROI")]
        return summary, panels

    def clear():
        return None, None, "Upload a dermoscopic lesion image, then select Analyse.", []

    with gr.Blocks(title="Attention U-Net melanoma research demo",
                   delete_cache=(3600, 3600)) as demo:
        gr.Markdown("# Melanoma image analysis\n"
                    "Attention U-Net lesion segmentation with EfficientNetB0 classification.\n\n"
                    "**Student research prototype. Not validated for clinical diagnosis.**")
        with gr.Tab("Analyse image"):
            with gr.Row():
                with gr.Column(scale=1):
                    upload = gr.File(label="Upload dermoscopic image (JPG or PNG)",
                                     file_types=[".jpg", ".jpeg", ".png"], type="filepath")
                    input_preview = gr.Image(label="Uploaded image", interactive=False)
                    with gr.Row():
                        analyse_button = gr.Button("Analyse", variant="primary")
                        clear_button = gr.Button("Clear")
                    if sample_paths:
                        gr.Examples(examples=[[str(p)] for p in sample_paths],
                                    inputs=upload, label="Public ISIC example images")
                with gr.Column(scale=2):
                    summary = gr.Markdown("Upload a dermoscopic lesion image, then select Analyse.")
                    panels = gr.Gallery(label="Pipeline outputs", columns=3,
                                        object_fit="contain", height="auto")
                    gr.Markdown("The lesion mask estimates the lesion boundary. "
                                "Grad-CAM shows image regions contributing to the model prediction; "
                                "it is not a lesion mask.")
            upload.change(preview, inputs=upload, outputs=input_preview)
            analyse_button.click(analyse, inputs=upload, outputs=[summary, panels],
                                 api_name="analyse", concurrency_limit=1)
            clear_button.click(clear, outputs=[upload, input_preview, summary, panels])
        with gr.Tab("Evaluation"):
            gr.Markdown("## Saved evaluation results\n"
                        "Classification and segmentation measure different tasks. "
                        "The official ISIC test set was viewed in earlier experiments; "
                        "these results are exploratory follow-up.")
            run_dir = pipeline.run_dir
            displayed = False
            if run_dir:
                for filename, label in [("complete_segmentation_test_metrics.csv", "Lesion segmentation"),
                                        ("complete_classification_test_metrics.csv", "Melanoma classification")]:
                    path = run_dir / "tables" / filename
                    if path.is_file():
                        gr.Markdown("### " + label)
                        gr.Dataframe(value=pd.read_csv(path).T.reset_index(),
                                     headers=["Metric", "Value"], interactive=False)
                        displayed = True
                figures = sorted((run_dir / "figures").glob("*.png"))
                if figures:
                    gr.Gallery(value=[(str(p), p.stem.replace("_", " ")) for p in figures],
                               label="Evaluation figures", columns=1, object_fit="contain")
                    displayed = True
            if not displayed:
                gr.Markdown("Run the reporting notebook to add test loss, metrics and evaluation figures.")
    return demo
