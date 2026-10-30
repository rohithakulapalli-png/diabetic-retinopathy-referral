"""Demo web app: upload a fundus photo, get a referral suggestion, a DR grade and a Grad-CAM heatmap.

    python app.py --run-dir runs/b4_512 --quality quality_thresholds.json           # local only
    python app.py --run-dir runs/b4_512 --quality quality_thresholds.json --share   # temporary public link

Research demo only. It is not a medical device and must not be used for clinical decisions.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import gradio as gr
import torch

from dr.explain import gradcam, overlay, predict_one, to_tensor
from dr.metrics import scores_to_grades
from dr.model import load_checkpoint
from dr.preprocess import crop_fundus
from dr.quality import assess, quality_metrics

GRADE_NAMES = ["No DR", "Mild NPDR", "Moderate NPDR", "Severe NPDR", "Proliferative DR"]
DISCLAIMER = ("*Research demo only. It is not a medical device and must not be used to make "
              "decisions about anyone's care.*")


def make_analyser(run_dir: Path, quality_path: Path | None, thresholds_path: Path | None = None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ckpt = load_checkpoint(run_dir / "best.pt", device)
    cfg = json.loads((thresholds_path or run_dir / "thresholds.json").read_text())
    size, ben = ckpt["args"]["img_size"], not ckpt["args"]["no_ben"]
    fov = ckpt["args"].get("fov_frac", 1.0)
    limits = json.loads(quality_path.read_text())["limits"] if quality_path else {}

    def analyse(image):
        if image is None:
            return None, None, "Upload a colour fundus photo, then press **Analyse**."
        crop = crop_fundus(image, size)
        if crop is None:
            return None, None, ("## Ungradable photo\nNo retina could be found in this image. "
                                "Retake the photo, or refer for an in-person eye exam.\n\n" + DISCLAIMER)
        issues = assess(quality_metrics(crop), limits) if limits else []
        x = to_tensor(crop, ben, fov).to(device)
        score, prob = predict_one(model, x, tta=cfg.get("tta", True))
        heat = overlay(crop, gradcam(model, x))
        grade = GRADE_NAMES[int(scores_to_grades([score], cfg["grade_thresholds"])[0])]
        if issues:
            headline = "## Photo quality too low for a reliable result"
            detail = ("Problems found: " + "; ".join(issues) + ". Retake the photo if possible, "
                      "otherwise refer for an in-person exam. The model output below is shown for "
                      "research only.")
        elif prob >= cfg["referable_threshold"]:
            headline = "## Refer to an eye doctor"
            detail = "The model found signs of referable diabetic retinopathy (moderate or worse)."
        else:
            headline = "## No referable DR found"
            detail = "The model did not find moderate or worse diabetic retinopathy in this photo."
        report = (f"{headline}\n{detail}\n\n"
                  "| | |\n|---|---|\n"
                  f"| Referral score | {prob:.2f} (refer at {cfg['referable_threshold']:.2f} or above) |\n"
                  f"| Estimated grade | {grade} |\n"
                  f"| Photo quality | {'flagged: ' + '; '.join(issues) if issues else 'OK'} |\n\n"
                  "The heatmap shows the areas that most raised the referral score. It is a rough "
                  "guide to where the model looked, not a map of lesions. When the referral score is low there is little signal, so the heatmap mostly shows noise.\n\n" + DISCLAIMER)
        return crop, heat, report

    return analyse


def build_ui(analyse) -> gr.Blocks:
    with gr.Blocks(title="DR screening demo") as demo:
        gr.Markdown("# Diabetic retinopathy screening demo\n" + DISCLAIMER)
        with gr.Row():
            with gr.Column():
                photo = gr.Image(type="numpy", image_mode="RGB", label="Fundus photo")
                button = gr.Button("Analyse", variant="primary")
            with gr.Column():
                report = gr.Markdown()
        with gr.Row():
            cropped = gr.Image(label="Cropped photo", interactive=False)
            heatmap = gr.Image(label="Grad-CAM heatmap", interactive=False)
        button.click(analyse, inputs=photo, outputs=[cropped, heatmap, report])
    return demo


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", required=True, help="folder with best.pt and thresholds.json")
    p.add_argument("--quality", help="quality_thresholds.json from calibrate_quality.py")
    p.add_argument("--thresholds", help="thresholds file for this clinic (from site_threshold.py)")
    p.add_argument("--share", action="store_true", help="create a temporary public link")
    p.add_argument("--port", type=int, default=7860)
    args = p.parse_args()
    analyse = make_analyser(Path(args.run_dir), Path(args.quality) if args.quality else None,
                            Path(args.thresholds) if args.thresholds else None)
    build_ui(analyse).launch(share=args.share, server_port=args.port)


if __name__ == "__main__":
    main()
