"""Gradio demo: screening decision, stage probabilities, preprocessed photo and Grad-CAM for one fundus photo."""

from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path

try:
    import spaces  # Hugging Face ZeroGPU, only installed there
except ImportError:
    spaces = None
import cv2
import gradio as gr
import numpy as np
import pandas as pd
from PIL import Image

from dr import CLASSES
from dr.data import eval_tfms
from dr.evaluate import gradcam, load_checkpoint, overlay_cam, raw_outputs, values_to_probs
from dr.model import DEFAULT_THRESHOLDS, outputs_to_grades
from dr.preprocess import preprocess, preprocess_v2, preprocess_v3, preprocess_v4
from dr.train import get_device

SAMPLES = Path("sample_images")
FINAL = Path("runs/final")

ADVICE = {
    0: "No diabetic retinopathy detected. Routine annual screening is still recommended.",
    1: "Mild non-proliferative changes. Usually monitored; recheck within 12 months.",
    2: "Moderate non-proliferative changes. Referral to an ophthalmologist is advised.",
    3: "Severe non-proliferative changes. Prompt specialist referral is advised.",
    4: "Proliferative diabetic retinopathy. Urgent specialist referral is advised.",
}

DISCLAIMER = (
    "Research prototype for a university coursework project. "
    "Not a medical device and not a substitute for examination by a clinician."
)

UNCERTAIN_BELOW = 0.50  # top probability below this -> "uncertain, refer"
SEVERE_ALERT_ABOVE = 0.20  # P(severe or proliferative) at or above this -> urgent

# photo-quality limits, calibrated on 1,200 photos from all four datasets (just outside the real range)
QUALITY = {"max_colour_ratio": 1.00, "min_lit_share": 0.05,
           "min_brightness": 25.0, "max_brightness": 220.0, "min_sharpness": 4.0}

LEVELS = {  # headline, background, border, text
    "none":      ("NO REFERRAL NEEDED", "#e8f6ee", "#2e7d4f", "#14532d"),
    "monitor":   ("MONITOR: RECHECK IN 12 MONTHS", "#fff8db", "#c9a400", "#5c4a00"),
    "refer":     ("REFER TO AN OPHTHALMOLOGIST", "#fff0e0", "#e07b00", "#6b3500"),
    "urgent":    ("URGENT REFERRAL", "#fde8e8", "#c62828", "#7f1d1d"),
    "uncertain": ("UNCERTAIN: REFER TO A SPECIALIST", "#eef0f3", "#6b7280", "#1f2937"),
    "retake":    ("RETAKE THE PHOTO", "#eef0f3", "#6b7280", "#1f2937"),
    "reject":    ("NOT A RETINAL PHOTOGRAPH", "#eef0f3", "#374151", "#111827"),
}


# quality check
def quality_check(img: np.ndarray) -> tuple[bool, list[str]]:
    """Returns (is a colour fundus photo?, quality warnings) for a raw RGB image."""
    grey = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    lit = grey > 7
    if lit.mean() < QUALITY["min_lit_share"]:
        return False, ["The image is almost entirely black."]

    r, g, b = img[lit].astype(float).mean(axis=0) + 1e-6
    if max(g / r, b / r) >= QUALITY["max_colour_ratio"]:
        return False, ["Red is not the dominant colour, as it is in every fundus photograph."]

    warnings = []
    brightness = float(grey[lit].mean())
    if brightness < QUALITY["min_brightness"]:
        warnings.append("Photo is very dark.")
    elif brightness > QUALITY["max_brightness"]:
        warnings.append("Photo is over-exposed.")

    v3 = preprocess_v3(img, 512)
    inner = cv2.erode((cv2.cvtColor(v3, cv2.COLOR_RGB2GRAY) > 7).astype(np.uint8), np.ones((25, 25), np.uint8)) > 0
    sharpness = float(cv2.Laplacian(v3[:, :, 1].astype(float), cv2.CV_64F)[inner].var()) if inner.any() else 0.0
    if sharpness < QUALITY["min_sharpness"]:
        warnings.append("Photo is blurred or out of focus.")
    return True, warnings


# decision
def decision(probs: np.ndarray, warnings: list[str]) -> dict:
    """Screening decision from the five probabilities and any quality warnings."""
    grade = int(probs.argmax())
    confidence = float(probs[grade])
    p_severe = float(probs[3] + probs[4])

    uncertain = confidence < UNCERTAIN_BELOW
    if uncertain:
        level = "uncertain"
    else:
        level = {0: "none", 1: "monitor", 2: "refer", 3: "urgent", 4: "urgent"}[grade]

    notes = []
    if p_severe >= SEVERE_ALERT_ABOVE and level != "urgent":
        level = "urgent"
        notes.append(f"{p_severe:.0%} chance of severe or proliferative disease: treat as urgent "
                     f"until a clinician has reviewed the image.")
    if warnings and level == "none":
        # a poor photo can never clear a patient
        level = "retake"
        notes.append("The model found no disease, but a healthy result cannot be trusted on a "
                     "poor-quality photo.")

    stage = f"Stage {grade}: {CLASSES[grade]} ({confidence:.0%} confidence)"
    if uncertain:
        stage = f"Closest stage {grade}: {CLASSES[grade]}, but only {confidence:.0%} confident"
    return {"level": level, "grade": grade, "confidence": confidence, "stage": stage,
            "advice": ADVICE[grade] if not uncertain and level != "retake" else "",
            "notes": notes, "warnings": warnings}


def verdict_text(probs: np.ndarray, detail: str = "") -> str:
    """Plain-text version of the decision (self-check and logs)."""
    d = decision(probs, [])
    lines = [LEVELS[d["level"]][0], d["stage"] + (f" [{detail}]" if detail else "")]
    lines += [d["advice"]] if d["advice"] else []
    lines += [f"ALERT: {n}" for n in d["notes"]]
    lines.append(DISCLAIMER)
    return "\n".join(line for line in lines if line)


def result_card(d: dict | None) -> str:
    """Colour-coded HTML card for a decision, or a neutral prompt when there is none."""
    if d is None:
        return ("<div class='dr-card' style='--card-fg:#374151;padding:18px;border:2px dashed #9ca3af;"
                "border-radius:12px;background:#f9fafb;font-size:15px'>Upload a fundus photograph, "
                "or click an example on the left, to get a screening result.</div>")
    headline, bg, border, fg = LEVELS[d["level"]]
    parts = [f"<div style='font-size:24px;font-weight:700;letter-spacing:.3px'>{headline}</div>"]
    if d.get("stage"):
        parts.append(f"<div style='font-size:16px;margin-top:6px'>{html.escape(d['stage'])}</div>")
    if d.get("advice"):
        parts.append(f"<div style='margin-top:10px'>{html.escape(d['advice'])}</div>")
    for note in d.get("notes", []):
        parts.append(f"<div style='margin-top:10px;font-weight:600'>&#9888; {html.escape(note)}</div>")
    for w in d.get("warnings", []):
        parts.append(f"<div style='margin-top:8px'>Photo quality: {html.escape(w)}</div>")
    return (f"<div class='dr-card' style='--card-fg:{fg};padding:18px 20px;border-radius:12px;"
            f"border:3px solid {border};background:{bg};font-size:15px;line-height:1.45'>"
            f"{''.join(parts)}</div>")


# model
def load(checkpoint: str | Path):
    """Load the checkpoint once at startup."""
    device = get_device()
    model, cfg = load_checkpoint(checkpoint, "cpu")
    return model.to(device), cfg, device, eval_tfms(cfg.get("img_size", 224))


def build_predict_fn(model, device, tfms, thresholds=DEFAULT_THRESHOLDS, tta: bool = True,
                     pipeline: str = "v1"):
    """Build the per-photo pipeline: quality check, the checkpoint's preprocessing, model, Grad-CAM."""
    prep = {"v4": lambda im: preprocess_v4(im, 512), "v3": lambda im: preprocess_v3(im, 512),
            "v2": lambda im: preprocess_v2(im, 512)}.get(pipeline, preprocess)

    def analyse(image: np.ndarray | None) -> dict:
        if image is None:
            return {"decision": None, "confidences": {}, "cleaned": None, "heat": None}
        image = np.ascontiguousarray(image[:, :, :3])
        is_fundus, warnings = quality_check(image)
        if not is_fundus:
            d = {"level": "reject", "stage": "", "notes": [],
                 "advice": "Upload a colour photograph of the retina taken with a fundus camera.",
                 "warnings": warnings}
            return {"decision": d, "confidences": {}, "cleaned": None, "heat": None}

        cleaned = prep(image)
        tensor = tfms(Image.fromarray(cleaned))
        raw = raw_outputs(model, tensor.unsqueeze(0).to(device), tta=tta).cpu()
        grade = int(outputs_to_grades(raw, thresholds)[0])
        probs = values_to_probs([float(raw[0, 0])])[0] if raw.shape[1] == 1 else raw[0].numpy()

        cam = gradcam(model, tensor, class_idx=grade, device=device)
        return {"decision": decision(np.asarray(probs), warnings),
                "confidences": {f"{i} {name}": float(probs[i]) for i, name in enumerate(CLASSES)},
                "cleaned": cleaned, "heat": overlay_cam(cleaned, cam)}

    return analyse


# page text
HOW_TO_READ = """
**The coloured card is the screening decision.**

| Colour | Meaning |
|---|---|
| Green | No diabetic retinopathy found. Routine annual screening |
| Yellow | Mild changes. Monitor and recheck within 12 months |
| Amber | Moderate changes. Refer to an ophthalmologist |
| Red | Severe or proliferative, or a 20 % or greater chance of either. Urgent referral |
| Grey | Uncertain (under 50 % confident), poor photo, or not a retinal photo |

**Stage probabilities** are the model's belief across the five grades; they add up to 100 %.

**What the model sees** is your photo after the same preprocessing used in training:
the black border cropped, and every photo cut to one standard outline so the model
cannot guess from the camera's frame shape.

**Grad-CAM** colours the regions that most influenced the decision, red strongest.
Heat on lesions (red dots, bleeds, yellow deposits) is reassuring. It shows where
the model looked, not why.

**Safety rules:** below 50 % confidence the page will not give a grade on its own;
any 20 % or greater chance of severe disease escalates to urgent; a photo that
fails the quality check can never be cleared as healthy.
"""


def model_card(cfg: dict) -> str:
    """Markdown for the "About the model" tab."""
    lines = [
        "## About the model",
        "",
        f"- **Network:** {cfg.get('arch', '?')}, a convolutional neural network pretrained on ImageNet, "
        "fine-tuned in two phases (head first, then the deepest three stages).",
        f"- **Input:** {cfg.get('img_size', '?')} px, preprocessing pipeline {cfg.get('pipeline', 'v1')} "
        "(crop, common outline, pad, resize; no enhancement), four-view test-time augmentation.",
        "- **Trained on:** " + ("10,999 cleaned photos from APTOS (India) and DDR (China), balanced by dataset "
                                "and grade" if "DDR" in cfg.get("note", "") else cfg.get("note", "")) +
        ". Duplicate and contradictory photos removed first.",
        "- **Intended use:** research demonstration of automated DR screening. Not a medical device.",
    ]
    metrics = FINAL / "metrics.csv"
    if metrics.exists():
        m = pd.read_csv(metrics)
        lines += ["", "### Test results (photos the model never trained on)", "",
                  "| Hospital | Photos | Kappa | Sick eyes caught | Healthy eyes cleared |",
                  "|---|---|---|---|---|"]
        for _, r in m.iterrows():
            lines.append(f"| {r['hospital']} | {int(r['photos']):,} | {r['quadratic kappa']:.3f} | "
                         f"{r['sensitivity (any DR)']:.0%} | {r['specificity (any DR)']:.0%} |")
        lines += ["", "APTOS and DDR test photos come from the same hospitals as the training photos; "
                      "IDRiD and Messidor-2 were never seen during training."]
    lines += [
        "", "### Known limitations", "",
        "- Accuracy drops at hospitals and cameras the model never saw (Messidor-2 is the weakest).",
        "- The doctors' labels themselves disagree on some photos, which caps any model.",
        "- Severe (grade 3) disease is the hardest stage; mild changes are often graded one step off.",
        "- The quality check is a simple rule: a reddish image that is not a retina could still pass.",
        "- Grad-CAM is a visual aid, not an explanation of clinical reasoning.",
    ]
    return "\n".join(lines)


CSS = """
.gradio-container {max-width: 1280px !important; margin: auto;}
#title h1 {margin-bottom: 0;}
/* keep the card text colour in both light and dark themes */
#result-card .dr-card, #result-card .dr-card * {color: var(--card-fg) !important;}
"""


# app
def build_ui(analyse, cfg: dict) -> gr.Blocks:
    def run(image):
        out = analyse(image)
        return result_card(out["decision"]), out["confidences"], out["cleaned"], out["heat"]

    samples = sorted(SAMPLES.glob("*.png")) if SAMPLES.exists() else []
    names = ["No DR", "Mild", "Moderate", "Severe", "Proliferative"]
    labels = [f"Grade {p.name[5]}: {names[int(p.name[5])]} #{p.stem[-1]}" for p in samples]

    with gr.Blocks(title="DR screening prototype") as demo:
        gr.Markdown("# Diabetic retinopathy screening prototype", elem_id="title")
        gr.Markdown("Upload a retinal fundus photograph, or click an example. "
                    "The model grades it on the five-stage clinical scale and gives a screening decision.")
        with gr.Tabs():
            with gr.Tab("Grade a photo"):
                with gr.Row(equal_height=False):
                    with gr.Column(scale=5):
                        image = gr.Image(type="numpy", label="Fundus photograph", height=380,
                                         sources=["upload", "clipboard"])
                        with gr.Row():
                            clear = gr.Button("Clear")
                            submit = gr.Button("Grade photo", variant="primary")
                    with gr.Column(scale=6):
                        card = gr.HTML(result_card(None), elem_id="result-card")
                        probs = gr.Label(num_top_classes=5, label="Stage probabilities")
                with gr.Row():
                    seen = gr.Image(label="What the model sees (after preprocessing)", height=340,
                                    interactive=False)
                    heat = gr.Image(label="Grad-CAM: where the model looked", height=340, interactive=False)
                outputs = [card, probs, seen, heat]
                if samples:
                    gr.Examples(examples=[[str(p)] for p in samples], inputs=[image], outputs=outputs,
                                fn=run, run_on_click=True, cache_examples=False, example_labels=labels,
                                examples_per_page=10, label="Example photos (sealed test photos; true grade shown)")
                with gr.Accordion("How to read this result", open=False):
                    gr.Markdown(HOW_TO_READ)
            with gr.Tab("About the model"):
                gr.Markdown(model_card(cfg))
        gr.Markdown(f"<sub>{DISCLAIMER}</sub>")

        submit.click(run, inputs=image, outputs=outputs)
        image.upload(run, inputs=image, outputs=outputs)
        clear.click(lambda: (None, result_card(None), {}, None, None), outputs=[image] + outputs)
    return demo


def main() -> None:
    parser = argparse.ArgumentParser(description="Diabetic retinopathy screening prototype")
    parser.add_argument("--checkpoint", default="runs/best/best.pt", help="path to best.pt")
    parser.add_argument("--share", action="store_true", help="create a public Gradio link")
    args = parser.parse_args()

    if not Path(args.checkpoint).exists():
        raise SystemExit(f"No checkpoint at {args.checkpoint}. Train a model first, "
                         "or pass --checkpoint path/to/best.pt")

    model, cfg, device, tfms = load(args.checkpoint)
    analyse = build_predict_fn(model, device, tfms, cfg.get("thresholds", DEFAULT_THRESHOLDS),
                               cfg.get("tta", True), cfg.get("pipeline", "v1"))
    if spaces:
        analyse = spaces.GPU(analyse)  # borrow a GPU per photo
    print(f"loaded {cfg['arch']} (pipeline {cfg.get('pipeline', 'v1')}, {cfg.get('img_size', 224)} px) "
          f"on {device}, tta={cfg.get('tta', True)}")
    build_ui(analyse, cfg).launch(share=args.share, theme=gr.themes.Soft(), css=CSS)


def self_check() -> None:
    import matplotlib

    from dr.preprocess import read_rgb

    # decision logic
    assert decision(np.array([0.93, .03, .02, .01, .01]), [])["level"] == "none"
    assert decision(np.array([.02, .9, .05, .02, .01]), [])["level"] == "monitor"
    assert decision(np.array([.05, .05, .8, .05, .05]), [])["level"] == "refer"
    assert decision(np.array([.02, .03, .05, .1, .8]), [])["level"] == "urgent"
    miss = decision(np.array([.09, .38, .23, .08, .22]), [])  # a real proliferative miss
    assert miss["level"] == "urgent" and miss["notes"], "a 30 % severe chance must escalate"
    assert "only 38% confident" in miss["stage"], "an escalated uncertain result must still say it is unsure"
    assert decision(np.array([.45, .4, .1, .03, .02]), [])["level"] == "uncertain"
    assert decision(np.array([.93, .03, .02, .01, .01]), ["Photo is blurred."])["level"] == "retake"
    assert "URGENT" in verdict_text(np.array([.09, .38, .23, .08, .22]))

    # quality check on real and non-fundus images
    fundus = read_rgb(SAMPLES / "grade2_moderate_1.png")
    assert quality_check(fundus) == (True, []), quality_check(fundus)
    portrait = read_rgb(Path(matplotlib.get_data_path()) / "sample_data" / "grace_hopper.jpg")
    assert quality_check(portrait)[0] is False, "a portrait must be rejected"
    assert quality_check(np.zeros((300, 300, 3), np.uint8))[0] is False
    assert quality_check((fundus * 0.12).astype(np.uint8))[1], "a very dark photo must be flagged"
    assert quality_check(cv2.GaussianBlur(fundus, (0, 0), 12))[1], "a blurred photo must be flagged"
    for p in sorted(SAMPLES.glob("*.png")):
        ok, w = quality_check(read_rgb(p))
        assert ok and not w, f"{p.name} is a real test photo and must pass: {w}"

    assert "NO REFERRAL" in result_card(decision(np.array([0.93, .03, .02, .01, .01]), []))
    print("app.py self-check passed")


if __name__ == "__main__":
    if "--self-check" in sys.argv:
        self_check()
    else:
        main()
