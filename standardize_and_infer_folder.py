#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Location: /users/kdm/divjots2002/gazemain/standardize_and_infer_folder.py

import os, glob, math, argparse, json
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from PIL import Image, ImageOps, ImageDraw, ImageFont

import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.models as models

# ---------------- utils ----------------

def wrap_deg(x: float) -> float:
    return (x + 180.0) % 360.0 - 180.0

def pil_fix_exif(pil: Image.Image) -> Image.Image:
    # Rotate image to upright if EXIF says so
    try:
        return ImageOps.exif_transpose(pil)
    except Exception:
        return pil

def centerface_crop(pil: Image.Image, frac: float = 0.75) -> Image.Image:
    """Fallback crop if no detector; center-crop to a face-like box."""
    w, h = pil.size
    side = int(min(w, h) * frac)
    cx, cy = w // 2, int(h * 0.45)  # bias slightly up toward eyes
    x1 = max(0, cx - side // 2); y1 = max(0, cy - side // 2)
    x2 = min(w, x1 + side);       y2 = min(h, y1 + side)
    return pil.crop((x1, y1, x2, y2))

def try_cv2_face_crop(pil: Image.Image) -> Tuple[Image.Image, Optional[float]]:
    """Use OpenCV Haar to detect face (and eyes for roll). Returns (crop, roll_deg or None)."""
    try:
        import cv2
        cv = cv2
    except Exception:
        return centerface_crop(pil), None

    img = np.array(pil.convert("RGB"))[:, :, ::-1]  # to BGR
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # Haar cascades
    haar_face = cv.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv.CascadeClassifier(haar_face)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5,
                                          flags=cv.CASCADE_SCALE_IMAGE, minSize=(60,60))
    if len(faces) == 0:
        return centerface_crop(pil), None

    # pick largest face
    x, y, w, h = max(faces, key=lambda b: b[2]*b[3])
    # expand a bit
    pad = int(0.2 * max(w, h))
    x1 = max(0, x - pad); y1 = max(0, y - pad)
    x2 = min(img.shape[1], x + w + pad); y2 = min(img.shape[0], y + h + pad)
    crop = Image.fromarray(img[y1:y2, x1:x2, ::-1], mode="RGB")

    # try eyes for roll
    roll_deg = None
    try:
        haar_eye = cv.data.haarcascades + "haarcascade_eye.xml"
        eye_cascade = cv.CascadeClassifier(haar_eye)
        roi_gray = gray[y1:y2, x1:x2]
        eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=5,
                                            flags=cv.CASCADE_SCALE_IMAGE, minSize=(20,20))
        # pick two widest eyes
        if len(eyes) >= 2:
            eyes_sorted = sorted(eyes, key=lambda e: e[2], reverse=True)[:2]
            # centers
            e_centers = []
            for (ex,ey,ew,eh) in eyes_sorted:
                e_centers.append((ex + ew/2.0, ey + eh/2.0))
            (xA, yA), (xB, yB) = e_centers[0], e_centers[1]
            # ensure left-right ordering
            if xA > xB:
                (xA, yA), (xB, yB) = (xB, yB), (xA, yA)
            # roll angle: negative if right eye is lower (clockwise tilt)
            roll_rad = math.atan2((yB - yA), (xB - xA + 1e-6))
            roll_deg = roll_rad * 180.0 / math.pi
    except Exception:
        pass

    return crop, roll_deg

def rotate_about_center(pil: Image.Image, deg: float) -> Image.Image:
    # expand=False to keep similar FOV; BICUBIC for quality
    return pil.rotate(deg, resample=Image.BICUBIC, expand=False)

def draw_gaze_overlay(pil_img: Image.Image,
                      yaw_ui_deg: float,
                      pitch_ui_deg: float,
                      note: str = "") -> Image.Image:
    img = pil_img.convert("RGB")
    w, h = img.size
    cx, cy = w // 2, h // 2
    axis_len = int(min(w, h) * 0.25)

    dr = ImageDraw.Draw(img)
    # axes
    dr.line([(cx - axis_len, cy), (cx + axis_len, cy)], fill=(255,0,0), width=4)    # +yaw right
    dr.line([(cx, cy + axis_len), (cx, cy - axis_len)], fill=(0,128,255), width=4)  # +pitch up

    # arrow (scale: 90° to end)
    scale = axis_len / 90.0
    dx = yaw_ui_deg * scale
    dy = -pitch_ui_deg * scale
    ex, ey = int(cx + dx), int(cy + dy)
    dr.line([(cx, cy), (ex, ey)], fill=(255,255,0), width=5)
    ang = math.atan2(ey - cy, ex - cx)
    ah = 16
    left  = (ex - ah * math.cos(ang - math.pi/6), ey - ah * math.sin(ang - math.pi/6))
    right = (ex - ah * math.cos(ang + math.pi/6), ey - ah * math.sin(ang + math.pi/6))
    dr.polygon([(ex, ey), left, right], fill=(255,255,0))

    # label
    try: font = ImageFont.truetype("DejaVuSans.ttf", 22)
    except: font = ImageFont.load_default()
    label = f"UI yaw={yaw_ui_deg:+.1f}°, pitch={pitch_ui_deg:+.1f}°  (+yaw=right, +pitch=up)"
    if note: label += f"  [{note}]"
    bbox = dr.textbbox((0,0), label, font=font)
    tw, th = bbox[2]-bbox[0], bbox[3]-bbox[1]
    pad = 8
    bg = Image.new("RGBA", (tw+2*pad, th+2*pad), (0,0,0,150))
    img_rgba = img.convert("RGBA")
    img_rgba.paste(bg, (10, 10), bg)
    ImageDraw.Draw(img_rgba).text((10+pad, 10+pad), label, fill=(255,255,255,255), font=font)
    return img_rgba.convert("RGB")

# --------------- model -----------------

class GazeResNet(nn.Module):
    """ResNet-18/50 with 2-dim head (pitch,yaw) in radians."""
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
        f = self.base(x); return self.head(f)

def load_ckpt_flex(model: nn.Module, ckpt_path: str):
    blob = torch.load(ckpt_path, map_location="cpu")
    if isinstance(blob, dict):
        sd = blob.get("state_dict", blob.get("model", blob))
        if hasattr(sd, "state_dict"): sd = sd.state_dict()
    else:
        sd = blob.state_dict() if hasattr(blob, "state_dict") else blob

    # normalize prefixes
    from collections import OrderedDict
    def strip_prefix(d, prefixes=("module.","model.","backbone.m.")):
        out = OrderedDict()
        for k,v in d.items():
            kk = k
            for p in prefixes:
                if kk.startswith(p): kk = kk[len(p):]
            out[kk] = v
        return out
    sd = strip_prefix(sd)

    # remap alternate heads
    remap = {}
    for k,v in sd.items():
        nk = k
        if k.startswith("gaze_head."): nk = "head." + k[len("gaze_head."):]
        if k.startswith("fc."):        nk = "head." + k[len("fc."):]
        remap[nk] = v
    sd = remap

    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"[ckpt] loaded: {ckpt_path}")
    print(f"[ckpt] missing={len(missing)} unexpected={len(unexpected)}")
    if missing:    print("  missing (first 10):", list(missing)[:10])
    if unexpected: print("  unexpected (first 10):", list(unexpected)[:10])

# --------------- transforms ------------

tf_face = T.Compose([
    T.Resize((224,224)),
    T.ToTensor(),
    T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

# --------------- main ------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images-dir", default="/users/kdm/divjots2002/gazemain/my_faces")
    ap.add_argument("--ckpt", default="/users/project1/pt01281/gaze_outputs/checkpoints_xgaze50/gaze_best.pth")
    ap.add_argument("--backbone", choices=["resnet18","resnet50"], default="resnet50")
    ap.add_argument("--outdir", default="/users/kdm/divjots2002/gazemain/my_gaze_outputs")
    ap.add_argument("--csv", default=None, help="CSV path (default: outdir/myfaces_standardized.csv)")
    # selfie / UI controls (LOCKED defaults that worked for you)
    ap.add_argument("--mirror-x", action="store_true", help="Unmirror typical webcam selfies (left<->right).")
    ap.add_argument("--flip-ui-pitch", action="store_true", help="Invert UI pitch sign after mapping.")
    ap.add_argument("--ui-roll-deg", type=float, default=None, help="Force extra UI roll (deg), applied after detection.")
    # calibration (optional)
    ap.add_argument("--affine-calib", default="/users/project1/pt01281/gaze_outputs/eth_eval_calib/affine_calibration_best.npz",
                    help="If exists: 2x2 A and 2-dim b (npz with keys A,b) applied to raw (pitch,yaw) radians.")
    args = ap.parse_args()

    img_glob = []
    for ext in ("*.jpg","*.jpeg","*.png","*.bmp","*.webp"):
        img_glob.extend(glob.glob(os.path.join(args.images_dir, ext)))
    files = sorted(img_glob, key=os.path.getmtime, reverse=True)
    if not files:
        print(f"[warn] no images under {args.images_dir}")
        return

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    csv_path = Path(args.csv) if args.csv else (outdir / "myfaces_standardized.csv")
    write_header = not csv_path.exists()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[info] found {len(files)} images")
    print(f"[info] device={device}")
    print(f"[info] loading checkpoint: {args.ckpt}")
    print(f"[info] backbone={args.backbone}")

    model = GazeResNet(args.backbone).to(device).eval()
    load_ckpt_flex(model, args.ckpt)

    # optional affine calibration
    A = b = None
    if args.affine_calib and os.path.exists(args.affine_calib):
        try:
            z = np.load(args.affine_calib)
            A = z["A"]; b = z["b"]
            if A.shape == (2,2) and b.shape == (2,):
                print(f"[calib] loaded affine A,b from {args.affine_calib}")
            else:
                print("[calib] bad shapes; ignoring")
                A = b = None
        except Exception as e:
            print("[calib] failed to load:", e)

    RAD2DEG = 180.0 / math.pi

    with open(csv_path, "a") as f:
        if write_header:
            f.write("path,mirror_x,flip_ui_pitch,roll_det_deg,roll_ui_deg,pr_pitch_rad,pr_yaw_rad,pr_pitch_deg,pr_yaw_deg,ui_yaw_deg,ui_pitch_deg\n")

        for img_path in files:
            try:
                raw = Image.open(img_path).convert("RGB")
                pil = pil_fix_exif(raw)

                # unmirror if requested
                note_bits = []
                if args.mirror_x:
                    pil = ImageOps.mirror(pil)
                    note_bits.append("unmirror")

                # detect face + roll, rotate to upright
                crop, roll_det = try_cv2_face_crop(pil)
                roll_ui = args.ui_roll_deg if args.ui_roll_deg is not None else 0.0
                roll_total = (roll_det or 0.0) + roll_ui
                if abs(roll_total) > 1e-3:
                    crop = rotate_about_center(crop, -roll_total)  # undo head roll (so axes align)

                x = tf_face(crop).unsqueeze(0).to(device)
                with torch.no_grad():
                    y = model(x)[0].detach().cpu().numpy()

                pr_pitch_rad = float(y[0]); pr_yaw_rad = float(y[1])
                # optional affine correction in radians (camera frame)
                if A is not None and b is not None:
                    P = np.array([pr_pitch_rad, pr_yaw_rad])
                    Pc = (P @ A) + b
                    pr_pitch_rad, pr_yaw_rad = float(Pc[0]), float(Pc[1])

                pr_pitch_deg = pr_pitch_rad * RAD2DEG
                pr_yaw_deg   = pr_yaw_rad   * RAD2DEG
                pr_pitch_deg_w = wrap_deg(pr_pitch_deg)
                pr_yaw_deg_w   = wrap_deg(pr_yaw_deg)

                # UI mapping: +yaw right, +pitch up
                yaw_ui_deg   = -pr_yaw_deg_w
                pitch_ui_deg = -pr_pitch_deg_w

                # your locked-in fix from yesterday:
                #  - we already un-mirrored the selfie above; no extra yaw flip needed here
                #  - your dataset needed pitch flip for display
                if args.flip_ui_pitch:
                    pitch_ui_deg = -pitch_ui_deg
                    note_bits.append("flip_ui_pitch")

                note = ", ".join(note_bits)
                overlay = draw_gaze_overlay(crop, yaw_ui_deg, pitch_ui_deg, note=note)
                out_img = outdir / (Path(img_path).stem + "_overlay.jpg")
                overlay.save(str(out_img), quality=92)

                print(f"[ok] {img_path}  ->  UI yaw={yaw_ui_deg:+.1f}  pitch={pitch_ui_deg:+.1f}   roll_det={roll_det if roll_det is not None else 0:.1f}  saved:{out_img.name}")

                f.write(",".join([
                    img_path,
                    "1" if args.mirror_x else "0",
                    "1" if args.flip_ui_pitch else "0",
                    f"{0.0 if roll_det is None else roll_det:.2f}",
                    f"{roll_ui:.2f}",
                    f"{pr_pitch_rad:.6f}", f"{pr_yaw_rad:.6f}",
                    f"{pr_pitch_deg:.2f}", f"{pr_yaw_deg:.2f}",
                    f"{yaw_ui_deg:.2f}", f"{pitch_ui_deg:.2f}",
                ]) + "\n")

            except Exception as e:
                print(f"[skip] {img_path}  err={e}")

if __name__ == "__main__":
    main()

