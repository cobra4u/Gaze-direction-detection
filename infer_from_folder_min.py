#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, glob, math, argparse, csv
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.models as models

def wrap_deg(x: float) -> float:
    return (x + 180.0) % 360.0 - 180.0

def draw_gaze_overlay(pil_img, yaw_ui_deg, pitch_ui_deg, axis_frac=0.25):
    img = pil_img.convert("RGB")
    w, h = img.size
    cx, cy = (w//2, h//2)
    L = int(min(w, h) * axis_frac)
    dr = ImageDraw.Draw(img)
    dr.line([(cx-L, cy), (cx+L, cy)], fill=(255,64,64), width=4)
    dr.line([(cx, cy+L), (cx, cy-L)], fill=(64,128,255), width=4)
    scale = L/90.0
    dx = yaw_ui_deg*scale; dy = -pitch_ui_deg*scale
    ex, ey = int(cx+dx), int(cy+dy)
    dr.line([(cx,cy),(ex,ey)], fill=(255,255,0), width=5)
    ang = math.atan2(ey-cy, ex-cx); ah=16
    left  = (ex - ah*math.cos(ang-math.pi/6), ey - ah*math.sin(ang-math.pi/6))
    right = (ex - ah*math.cos(ang+math.pi/6), ey - ah*math.sin(ang+math.pi/6))
    dr.polygon([(ex,ey), left, right], fill=(255,255,0))
    try: font = ImageFont.truetype("DejaVuSans.ttf", 22)
    except: font = ImageFont.load_default()
    label = f"UI: yaw={yaw_ui_deg:+.1f}°, pitch={pitch_ui_deg:+.1f}°"
    bbox = dr.textbbox((0,0), label, font=font); tw=bbox[2]-bbox[0]; th=bbox[3]-bbox[1]
    pad=8; bg=Image.new("RGBA",(tw+2*pad, th+2*pad),(0,0,0,150))
    img_rgba = img.convert("RGBA"); img_rgba.paste(bg,(10,10),bg)
    ImageDraw.Draw(img_rgba).text((10+pad,10+pad), label, fill=(255,255,255,255), font=font)
    return img_rgba.convert("RGB")

class GazeResNet(nn.Module):
    def __init__(self, backbone="resnet50"):
        super().__init__()
        if backbone == "resnet18":
            m = models.resnet18(weights=None); feat=512
        elif backbone == "resnet50":
            m = models.resnet50(weights=None); feat=2048
        else:
            raise ValueError
        m.fc = nn.Identity()
        self.base = m; self.head = nn.Linear(feat, 2)
    def forward(self, x): return self.head(self.base(x))

tf_face = T.Compose([
    T.Resize((224,224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
])

def list_images(folder: str):
    files=[]
    for e in ("*.jpg","*.jpeg","*.png","*.bmp","*.webp"):
        files += glob.glob(os.path.join(folder, e))
    return sorted(files)  # by name for reproducibility

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images-dir", default="/users/kdm/divjots2002/gazemain/my_faces")
    ap.add_argument("--ckpt", default="/users/project1/pt01281/gaze_outputs/checkpoints_xgaze50/gaze_best.pth")
    ap.add_argument("--backbone", choices=["resnet18","resnet50"], default="resnet50")
    ap.add_argument("--outdir", default="/users/kdm/divjots2002/gazemain/my_gaze_outputs")
    ap.add_argument("--outcsv", default=None)
    ap.add_argument("--save-overlays", action="store_true", default=True)
    ap.add_argument("--mirror-x", action="store_true", default=True)
    ap.add_argument("--flip-ui-yaw", action="store_true", default=False)
    ap.add_argument("--flip-ui-pitch", action="store_true", default=True)
    args = ap.parse_args()

    files = list_images(args.images_dir)
    if not files: 
        print(f"[error] no images in {args.images_dir}"); return
    print(f"[info] found {len(files)} images")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_grad_enabled(False)
    model = GazeResNet(args.backbone).to(device).eval()
    ckpt = torch.load(args.ckpt, map_location="cpu")
    if "state_dict" in ckpt: ckpt = ckpt["state_dict"]
    model.load_state_dict(ckpt, strict=False)

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    csv_path = Path(args.outcsv) if args.outcsv else (outdir / "myfaces_preds_min.csv")
    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="") as fcsv:
        w = csv.writer(fcsv)
        if write_header:
            w.writerow(["path","pr_pitch_deg","pr_yaw_deg","pr_pitch_deg_wrapped","pr_yaw_deg_wrapped",
                        "ui_yaw_deg","ui_pitch_deg","mirror_x","flip_ui_yaw","flip_ui_pitch"])
        for i, p in enumerate(files, 1):
            pil = Image.open(p).convert("RGB")   # NO rotation/deskew
            x = tf_face(pil).unsqueeze(0).to(device)
            y = model(x)[0].detach().cpu().numpy()
            pr_pitch_deg = float(y[0]) * 180.0/math.pi
            pr_yaw_deg   = float(y[1]) * 180.0/math.pi
            ppw = wrap_deg(pr_pitch_deg); pyw = wrap_deg(pr_yaw_deg)

            yaw_ui   = -pyw
            pitch_ui = -ppw
            if args.mirror_x:       yaw_ui = -yaw_ui
            if args.flip_ui_yaw:    yaw_ui = -yaw_ui
            if args.flip_ui_pitch:  pitch_ui = -pitch_ui

            if args.save_overlays:
                overlay = draw_gaze_overlay(pil, yaw_ui, pitch_ui)
                out_img = outdir / (Path(p).stem + "_overlay.jpg")
                overlay.save(str(out_img), quality=92)

            w.writerow([p, f"{pr_pitch_deg:.2f}", f"{pr_yaw_deg:.2f}", f"{ppw:.2f}", f"{pyw:.2f}",
                        f"{yaw_ui:.2f}", f"{pitch_ui:.2f}",
                        int(args.mirror_x), int(args.flip_ui_yaw), int(args.flip_ui_pitch)])
            if i % 20 == 0 or i == len(files):
                print(f"[info] processed {i}/{len(files)}")

    print(f"[done] wrote CSV: {csv_path}")
    if args.save_overlays:
        print(f"[done] overlays in: {outdir}")

if __name__ == "__main__":
    main()

