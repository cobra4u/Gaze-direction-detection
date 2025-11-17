#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, glob, math, argparse, csv
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.models as models

# -------- utils --------
def wrap_deg(x: float) -> float:
    return (x + 180.0) % 360.0 - 180.0

def draw_gaze_overlay(pil_img: Image.Image,
                      yaw_ui_deg: float,
                      pitch_ui_deg: float,
                      origin: Optional[Tuple[int,int]] = None,
                      axis_frac: float = 0.25) -> Image.Image:
    """Draw axes (+yaw right, +pitch up) and a yellow arrow from origin."""
    img = pil_img.convert("RGB")
    w, h = img.size
    cx, cy = origin if origin else (w // 2, h // 2)
    axis_len = int(min(w, h) * axis_frac)

    dr = ImageDraw.Draw(img)
    # axes: red = +yaw (right), blue = +pitch (up)
    dr.line([(cx - axis_len, cy), (cx + axis_len, cy)], fill=(255, 64, 64), width=4)      # horizontal
    dr.line([(cx, cy + axis_len), (cx, cy - axis_len)], fill=(64, 128, 255), width=4)     # vertical

    # arrow (scale so 90° reaches axis end)
    scale = axis_len / 90.0
    dx = yaw_ui_deg * scale
    dy = -pitch_ui_deg * scale  # image y is down
    ex, ey = int(cx + dx), int(cy + dy)
    dr.line([(cx, cy), (ex, ey)], fill=(255, 255, 0), width=5)

    # arrow head
    ang = math.atan2(ey - cy, ex - cx)
    ah = 16
    left  = (ex - ah * math.cos(ang - math.pi/6), ey - ah * math.sin(ang - math.pi/6))
    right = (ex - ah * math.cos(ang + math.pi/6), ey - ah * math.sin(ang + math.pi/6))
    dr.polygon([(ex, ey), left, right], fill=(255, 255, 0))

    # label
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 22)
    except:
        font = ImageFont.load_default()
    label = f"UI: yaw={yaw_ui_deg:+.1f}°, pitch={pitch_ui_deg:+.1f}°  (+yaw=right, +pitch=up)"
    bbox = dr.textbbox((0, 0), label, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    pad = 8
    bg = Image.new("RGBA", (tw + 2*pad, th + 2*pad), (0, 0, 0, 150))
    img_rgba = img.convert("RGBA")
    img_rgba.paste(bg, (10, 10), bg)
    ImageDraw.Draw(img_rgba).text((10+pad, 10+pad), label, fill=(255, 255, 255, 255), font=font)
    return img_rgba.convert("RGB")

# -------- model --------
class GazeResNet(nn.Module):
    def __init__(self, backbone="resnet50"):
        super().__init__()
        if backbone == "resnet18":
            m = models.resnet18(weights=None); feat = 512
        elif backbone == "resnet50":
            m = models.resnet50(weights=None); feat = 2048
        else:
            raise ValueError("backbone must be resnet18 or resnet50")
        m.fc = nn.Identity()
        self.base = m
        self.head = nn.Linear(feat, 2)

    def forward(self, x):
        return self.head(self.base(x))

tf_face = T.Compose([
    # IMPORTANT: no exif/autorotate, no deskew; just resize+normalize
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def pick_latest_image(folder: str) -> str:
    files = []
    for e in ("*.jpg","*.jpeg","*.png","*.bmp","*.webp"):
        files += glob.glob(os.path.join(folder, e))
    if not files:
        raise FileNotFoundError(f"No images found under {folder}")
    files = sorted(files, key=os.path.getmtime, reverse=True)
    return files[0]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default=None, help="Path to one image; if missing, newest in --images-dir")
    ap.add_argument("--images-dir", default="/users/kdm/divjots2002/gazemain/my_faces")
    ap.add_argument("--ckpt", default="/users/project1/pt01281/gaze_outputs/checkpoints_xgaze50/gaze_best.pth")
    ap.add_argument("--backbone", choices=["resnet18","resnet50"], default="resnet50")
    ap.add_argument("--outdir", default="/users/kdm/divjots2002/gazemain/my_gaze_outputs")
    ap.add_argument("--csv", default=None)
    # Fixed mapping you validated: selfie (mirror-x), and flip pitch
    ap.add_argument("--mirror-x", action="store_true", default=True, help="Treat input as mirrored selfie")
    ap.add_argument("--flip-ui-yaw", action="store_true", default=False)
    ap.add_argument("--flip-ui-pitch", action="store_true", default=True)
    args = ap.parse_args()

    img_path = args.image or pick_latest_image(args.images_dir)
    pil = Image.open(img_path).convert("RGB")         # NO auto-rotate, we keep raw orientation

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[info] device={device}")
    print(f"[info] ckpt={args.ckpt} backbone={args.backbone}")

    torch.set_grad_enabled(False)
    model = GazeResNet(args.backbone).to(device).eval()
    ckpt = torch.load(args.ckpt, map_location="cpu")
    if "state_dict" in ckpt: ckpt = ckpt["state_dict"]
    missing, unexpected = model.load_state_dict(ckpt, strict=False)
    if missing or unexpected:
        print(f"[ckpt] missing={len(missing)} unexpected={len(unexpected)}")

    x = tf_face(pil).unsqueeze(0).to(device)
    y = model(x)[0].detach().cpu().numpy()  # [pitch(rad), yaw(rad)]
    pr_pitch_deg = float(y[0]) * 180.0/math.pi
    pr_yaw_deg   = float(y[1]) * 180.0/math.pi
    pr_pitch_deg_w = wrap_deg(pr_pitch_deg)
    pr_yaw_deg_w   = wrap_deg(pr_yaw_deg)

    # camera->UI: +yaw right, +pitch up
    yaw_ui   = -pr_yaw_deg_w
    pitch_ui = -pr_pitch_deg_w
    # selfie: mirror flips horizontal only
    if args.mirror_x:
        yaw_ui = -yaw_ui
    # manual flips (safety knobs)
    if args.flip_ui_yaw:   yaw_ui   = -yaw_ui
    if args.flip_ui_pitch: pitch_ui = -pitch_ui

    print("\n=== Prediction ===")
    print(f"image: {img_path}")
    print(f"camera(deg): pitch={pr_pitch_deg:.2f}, yaw={pr_yaw_deg:.2f}")
    print(f"wrapped(deg): pitch={pr_pitch_deg_w:.2f}, yaw={pr_yaw_deg_w:.2f}")
    print(f"UI(deg): yaw={yaw_ui:+.1f}, pitch={pitch_ui:+.1f}"
          + (" [mirror-x]" if args.mirror_x else "")
          + (" [flip-yaw]" if args.flip_ui_yaw else "")
          + (" [flip-pitch]" if args.flip_ui_pitch else ""))

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    overlay = draw_gaze_overlay(pil, yaw_ui, pitch_ui)
    out_img = outdir / (Path(img_path).stem + "_overlay.jpg")
    overlay.save(str(out_img), quality=92)
    print(f"[saved] overlay -> {out_img}")

    csv_path = Path(args.csv) if args.csv else (outdir / "_demo_min.csv")
    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(["path","pr_pitch_deg","pr_yaw_deg","pr_pitch_deg_wrapped","pr_yaw_deg_wrapped",
                        "ui_yaw_deg","ui_pitch_deg","mirror_x","flip_ui_yaw","flip_ui_pitch"])
        w.writerow([img_path, f"{pr_pitch_deg:.2f}", f"{pr_yaw_deg:.2f}",
                    f"{pr_pitch_deg_w:.2f}", f"{pr_yaw_deg_w:.2f}",
                    f"{yaw_ui:.2f}", f"{pitch_ui:.2f}",
                    int(args.mirror_x), int(args.flip_ui_yaw), int(args.flip_ui_pitch)])
    print(f"[saved] csv -> {csv_path}")

if __name__ == "__main__":
    main()

