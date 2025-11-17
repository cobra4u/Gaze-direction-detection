#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys, glob, math, json, argparse
from pathlib import Path
from typing import Tuple, Optional, List

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Optional deps
try:
    import cv2  # OpenCV
except Exception:
    cv2 = None

try:
    import mediapipe as mp  # for landmarks/roll if available
    _HAS_MP = True
except Exception:
    mp = None
    _HAS_MP = False

import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.models as tv

# ----------------------------- utils ---------------------------------

RAD2DEG = 180.0 / math.pi
DEG2RAD = math.pi / 180.0

def wrap_deg(x: float) -> float:
    """Wrap degrees to [-180, 180)."""
    return (x + 180.0) % 360.0 - 180.0

def safe_imread(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")

def pil_to_cv(img: Image.Image) -> np.ndarray:
    # PIL RGB -> OpenCV BGR
    a = np.array(img)
    return a[:, :, ::-1].copy()

def cv_to_pil(a: np.ndarray) -> Image.Image:
    # OpenCV BGR -> PIL RGB
    return Image.fromarray(a[:, :, ::-1].copy())

def rotate_pil(img: Image.Image, angle_deg: float, expand=True) -> Image.Image:
    # positive angle_deg rotates counter-clockwise
    return img.rotate(angle_deg, resample=Image.BICUBIC, expand=expand)

def draw_gaze_overlay(pil_img: Image.Image,
                      yaw_ui_deg: float,
                      pitch_ui_deg: float,
                      text_prefix: str = "UI") -> Image.Image:
    """Draw axes and a gaze arrow (UI: +yaw right, +pitch up)."""
    img = pil_img.convert("RGB")
    w, h = img.size
    cx, cy = w // 2, h // 2
    axis_len = int(min(w, h) * 0.28)

    dr = ImageDraw.Draw(img)
    # axes
    dr.line([(cx - axis_len, cy), (cx + axis_len, cy)], fill=(220, 60, 60), width=5)   # x (+yaw right)
    dr.line([(cx, cy + axis_len), (cx, cy - axis_len)], fill=(70, 140, 255), width=5)  # y (+pitch up)

    # arrow (scale so 90deg touches axis end)
    scale = axis_len / 90.0
    dx = yaw_ui_deg * scale
    dy = -pitch_ui_deg * scale
    ex, ey = int(cx + dx), int(cy + dy)
    dr.line([(cx, cy), (ex, ey)], fill=(255, 230, 60), width=6)
    ang = math.atan2(ey - cy, ex - cx)
    ah = 18
    left  = (ex - ah * math.cos(ang - math.pi/6), ey - ah * math.sin(ang - math.pi/6))
    right = (ex - ah * math.cos(ang + math.pi/6), ey - ah * math.sin(ang + math.pi/6))
    dr.polygon([(ex, ey), left, right], fill=(255, 230, 60))

    # label
    label = f"{text_prefix} yaw={yaw_ui_deg:+.1f}°, pitch={pitch_ui_deg:+.1f}°  (+yaw=right, +pitch=up)"
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 24)
    except Exception:
        font = ImageFont.load_default()
    bbox = dr.textbbox((0,0), label, font=font)
    tw, th = bbox[2]-bbox[0], bbox[3]-bbox[1]
    pad = 8
    bg = Image.new("RGBA", (tw + 2*pad, th + 2*pad), (0,0,0,155))
    img_rgba = img.convert("RGBA")
    img_rgba.paste(bg, (12, 12), bg)
    ImageDraw.Draw(img_rgba).text((12+pad, 12+pad), label, fill=(255,255,255,255), font=font)
    return img_rgba.convert("RGB")

# ------------------------- detection / crop --------------------------

def detect_face_bbox(img: Image.Image) -> Optional[Tuple[int,int,int,int]]:
    """
    Try MediaPipe FaceDetection → OpenCV Haar → None.
    Returns (x,y,w,h) in original orientation.
    """
    if _HAS_MP:
        try:
            mp_fd = mp.solutions.face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)
            cv = pil_to_cv(img)
            cv_rgb = cv[:, :, ::-1]
            res = mp_fd.process(cv_rgb)
            mp_fd.close()
            if res.detections:
                det = res.detections[0]
                bb = det.location_data.relative_bounding_box
                h, w = cv.shape[:2]
                x = max(0, int(bb.xmin * w))
                y = max(0, int(bb.ymin * h))
                bw = max(1, int(bb.width * w))
                bh = max(1, int(bb.height * h))
                return (x, y, bw, bh)
        except Exception:
            pass

    if cv2 is not None:
        try:
            gray = cv2.cvtColor(pil_to_cv(img), cv2.COLOR_BGR2GRAY)
            # Try both default haar files
            for haar in ("haarcascade_frontalface_default.xml", "haarcascade_frontalface_alt2.xml"):
                path = cv2.data.haarcascades + haar
                clf = cv2.CascadeClassifier(path)
                faces = clf.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, flags=cv2.CASCADE_SCALE_IMAGE)
                if len(faces) > 0:
                    # pick biggest
                    faces = sorted(faces, key=lambda r: r[2]*r[3], reverse=True)
                    x,y,w,h = faces[0]
                    return (int(x),int(y),int(w),int(h))
        except Exception:
            pass

    return None

def estimate_roll_deg(img: Image.Image) -> float:
    """
    Try MediaPipe FaceMesh roll; else 0.
    Positive = head tilt CCW.
    """
    if not _HAS_MP:
        return 0.0
    try:
        mp_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=True, refine_landmarks=True, max_num_faces=1)
        cv = pil_to_cv(img)
        res = mp_mesh.process(cv[:, :, ::-1])
        mp_mesh.close()
        if not res.multi_face_landmarks:
            return 0.0
        lm = res.multi_face_landmarks[0].landmark
        h, w = cv.shape[:2]
        # approximate roll from outer eye corners (left eye idx 33, right eye idx 263 in FaceMesh)
        pts = []
        for idx in (33, 263):
            pt = lm[idx]
            pts.append((pt.x*w, pt.y*h))
        (x1,y1),(x2,y2) = pts
        ang = math.degrees(math.atan2((y2 - y1), (x2 - x1)))
        # if eyes are perfectly horizontal, ang≈0. Positive CCW.
        return ang
    except Exception:
        return 0.0

def crop_face_wide(img: Image.Image, bbox: Optional[Tuple[int,int,int,int]], margin: float=1.8) -> Image.Image:
    """
    Expand detected face bbox by 'margin' to include forehead/chin/sides.
    If bbox is None, fall back to central square crop covering ~70% of shorter side.
    """
    W, H = img.size
    if bbox is None:
        s = int(0.7 * min(W, H))
        x = (W - s)//2; y = (H - s)//2
        return img.crop((x,y,x+s,y+s))

    x,y,w,h = bbox
    cx = x + w/2.0
    cy = y + h/2.0
    s = int(max(w,h) * margin)
    # make square
    x0 = int(cx - s/2); y0 = int(cy - s/2)
    x1 = int(cx + s/2); y1 = int(cy + s/2)
    # clamp
    x0 = max(0, x0); y0 = max(0, y0)
    x1 = min(W, x1); y1 = min(H, y1)
    if x1 <= x0 or y1 <= y0:
        return img
    return img.crop((x0,y0,x1,y1))

# ----------------------------- models --------------------------------

class GazeResNet(nn.Module):
    """ResNet-18/50 + linear(2) head (pitch,yaw) in radians."""
    def __init__(self, backbone="resnet50"):
        super().__init__()
        if backbone == "resnet18":
            m = tv.resnet18(weights=None); feat = 512
        elif backbone == "resnet50":
            m = tv.resnet50(weights=None); feat = 2048
        else:
            raise ValueError("backbone must be resnet18 or resnet50")
        m.fc = nn.Identity()
        self.backbone = m
        self.head = nn.Linear(feat, 2)

    def forward(self, x):
        f = self.backbone(x)
        return self.head(f)

def load_ckpt_flex(model: nn.Module, ckpt_path: str):
    """
    Accepts many common layouts: state_dict / model / gaze_head / fc / head…
    """
    blob = torch.load(ckpt_path, map_location="cpu")
    if isinstance(blob, dict):
        for k in ("state_dict","model","weights","net"):
            if k in blob:
                blob = blob[k]
                if hasattr(blob, "state_dict"):
                    blob = blob.state_dict()
                break
    if hasattr(blob, "state_dict"):
        sd = blob.state_dict()
    else:
        sd = dict(blob)

    # strip prefixes
    clean = {}
    for k,v in sd.items():
        kk = k
        for p in ("module.","model.","backbone.m.", "m."):
            if kk.startswith(p):
                kk = kk[len(p):]
        # map gaze_head/fc to head.*
        if kk.startswith("gaze_head."):
            kk = "head." + kk[len("gaze_head."):]
        elif kk.startswith("fc."):
            kk = "head." + kk[3:]
        clean[kk] = v

    missing, unexpected = model.load_state_dict(clean, strict=False)
    print(f"[ckpt] loaded: {ckpt_path}")
    print(f"[ckpt] missing={len(missing)} unexpected={len(unexpected)}")
    if missing:    print("  missing (first 10):", list(missing)[:10])
    if unexpected: print("  unexpected (first 10):", list(unexpected)[:10])

# optional demographics (UTKFace) — pass --utk-ckpt to enable
class DemoHead(nn.Module):
    """Tiny MLP over backbone features -> (age_bin, gender, race)."""
    def __init__(self, feat=2048):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(feat, 512), nn.ReLU(True))
        self.age  = nn.Linear(512, 3)   # 18-34, 35-54, 55+
        self.gender = nn.Linear(512, 2) # Man/Woman
        self.race = nn.Linear(512, 5)   # rough bins

    def forward(self, feat):
        z = self.fc(feat)
        return self.age(z), self.gender(z), self.race(z)

# --------------------------- transforms ------------------------------

TF_ETH = T.Compose([
    T.Resize((224,224)),
    T.ToTensor(),
    T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

# ------------------------------ core ---------------------------------

def infer_on_pil(pil_img: Image.Image,
                 model: GazeResNet,
                 device: str,
                 mirror_x=False,
                 flip_ui_yaw=False,
                 flip_ui_pitch=True,
                 roll_deg_ui: float = 0.0,
                 return_overlay=True) -> Tuple[float,float, Optional[Image.Image]]:
    """
    Returns (yaw_ui_deg, pitch_ui_deg, overlay)
    """
    model.eval()
    with torch.no_grad():
        x = TF_ETH(pil_img).unsqueeze(0).to(device)
        y = model(x)[0].detach().cpu().numpy()

    pr_pitch_deg = float(y[0] * RAD2DEG)
    pr_yaw_deg   = float(y[1] * RAD2DEG)
    pr_pitch_deg_w = wrap_deg(pr_pitch_deg)
    pr_yaw_deg_w   = wrap_deg(pr_yaw_deg)

    # camera -> UI (+yaw right, +pitch up)
    yaw_ui   = -pr_yaw_deg_w
    pitch_ui = -pr_pitch_deg_w

    if mirror_x:
        yaw_ui = -yaw_ui
    if flip_ui_yaw:
        yaw_ui = -yaw_ui
    if flip_ui_pitch:
        pitch_ui = -pitch_ui
    if abs(roll_deg_ui) > 1e-3:
        # rotate the vector by -roll in UI plane so arrow follows the screen axes
        th = -roll_deg_ui * DEG2RAD
        cy, sy = math.cos(th), math.sin(th)
        x_ui = yaw_ui; y_ui = pitch_ui
        yaw_ui   =  x_ui*cy - y_ui*sy
        pitch_ui =  x_ui*sy + y_ui*cy

    overlay = draw_gaze_overlay(pil_img, yaw_ui, pitch_ui) if return_overlay else None
    return yaw_ui, pitch_ui, overlay

def pick_latest(images_dir: str) -> str:
    files = []
    for ext in ("*.jpg","*.jpeg","*.png","*.bmp","*.webp"):
        files += glob.glob(os.path.join(images_dir, ext))
    if not files:
        raise FileNotFoundError(f"No images in {images_dir}")
    files.sort(key=os.path.getmtime, reverse=True)
    return files[0]

# ----------------------------- script --------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default=None, help="Run on this image (else newest from --images-dir).")
    ap.add_argument("--images-dir", default="/users/kdm/divjots2002/gazemain/my_faces")
    ap.add_argument("--all", action="store_true", help="Process the whole folder instead of a single image.")
    ap.add_argument("--ckpt", required=True, help="ETH-XGaze checkpoint (.pth/.pt)")
    ap.add_argument("--backbone", choices=["resnet18","resnet50"], default="resnet50")
    ap.add_argument("--outdir", default="/users/kdm/divjots2002/gazemain/my_gaze_outputs")
    ap.add_argument("--csv", default=None, help="CSV path (default: <outdir>/myfaces_preds.csv)")
    ap.add_argument("--save-overlays", action="store_true")
    # UI knobs
    ap.add_argument("--mirror-x", action="store_true", help="If webcam selfies are mirrored.")
    ap.add_argument("--flip-ui-yaw", action="store_true")
    ap.add_argument("--flip-ui-pitch", action="store_true")
    ap.add_argument("--no-auto-deroll", action="store_true", help="Disable automatic roll estimation/derotation.")
    ap.add_argument("--force-roll-deg", type=float, default=None, help="Override roll (deg) applied to UI vectors.")
    # optional demographics (only if you pass a UTKFace-ish ckpt; otherwise ignored)
    ap.add_argument("--utk-ckpt", default=None, help="Optional checkpoint for demographics head (if you have one).")
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    csv_path = Path(args.csv) if args.csv else (outdir / "myfaces_preds.csv")
    write_header = not csv_path.exists()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = GazeResNet(args.backbone).to(device).eval()
    load_ckpt_flex(model, args.ckpt)

    if args.all:
        files = []
        for ext in ("*.jpg","*.jpeg","*.png","*.bmp","*.webp"):
            files += glob.glob(os.path.join(args.images_dir, ext))
        files.sort(key=os.path.getmtime, reverse=True)
    else:
        path = args.image or pick_latest(args.images_dir)
        files = [path]

    # prepare demographics if requested (best-effort; optional)
    demo_head = None
    if args.utk_ckpt:
        feat = 512 if args.backbone=="resnet18" else 2048
        demo_head = DemoHead(feat).to(device).eval()
        try:
            blob = torch.load(args.utk_ckpt, map_location="cpu")
            sd = blob["state_dict"] if isinstance(blob, dict) and "state_dict" in blob else blob
            if hasattr(sd, "state_dict"):
                sd = sd.state_dict()
            # naive load (best-effort)
            demo_head.load_state_dict(sd, strict=False)
            print("[utk] demographics head loaded")
        except Exception as e:
            print("[utk] load failed:", e)
            demo_head = None

    # writer
    import csv as _csv
    with open(csv_path, "a", newline="") as f:
        w = _csv.writer(f)
        if write_header:
            cols = ["path","ui_yaw_deg","ui_pitch_deg","mirror_x","flip_ui_yaw","flip_ui_pitch","roll_used_deg",
                    "pr_pitch_deg","pr_yaw_deg","pr_pitch_wrapped_deg","pr_yaw_wrapped_deg"]
            if demo_head is not None:
                cols += ["age_bin","gender","race"]
            w.writerow(cols)

        for p in files:
            pil0 = safe_imread(p)

            # 1) detect and deroll (estimate roll from full frame to avoid “nose zoom”)
            roll_deg = 0.0
            if not args.no_auto_deroll and _HAS_MP:
                roll_deg = estimate_roll_deg(pil0)
            if args.force_roll_deg is not None:
                roll_deg = float(args.force_roll_deg)

            pil_derolled = rotate_pil(pil0, -roll_deg, expand=True)  # remove roll visually

            # 2) detect face bbox on derolled image, then crop with wide margin
            bbox = detect_face_bbox(pil_derolled)
            crop = crop_face_wide(pil_derolled, bbox, margin=1.8)

            # 3) (Optional) horizontally mirror if selfie
            if args.mirror_x:
                crop = crop.transpose(Image.FLIP_LEFT_RIGHT)

            # 4) infer gaze
            yaw_ui, pitch_ui, overlay_core = infer_on_pil(
                crop, model, device,
                mirror_x=False,  # already applied to crop if requested
                flip_ui_yaw=args.flip_ui_yaw,
                flip_ui_pitch=args.flip_ui_pitch,
                roll_deg_ui=0.0,  # crop is already derolled visually
                return_overlay=True
            )

            # 5) overlay label on the *cropped, derolled* region (stable)
            overlay = draw_gaze_overlay(crop, yaw_ui, pitch_ui, text_prefix="UI")

            # 6) demographics (optional placeholder: returns dummy labels unless you supply a UTK ckpt)
            age_bin = gender = race = ""
            if demo_head is not None:
                with torch.no_grad():
                    feat = model.backbone(TF_ETH(crop).unsqueeze(0).to(device))
                    a,g,r = demo_head(feat)
                    age_idx = int(torch.argmax(a, dim=1).item())
                    gen_idx = int(torch.argmax(g, dim=1).item())
                    race_idx = int(torch.argmax(r, dim=1).item())
                    age_map = {0:"18-34",1:"35-54",2:"55+"}
                    gen_map = {0:"Man",1:"Woman"}
                    race_map = {0:"Asian",1:"Black",2:"White",3:"Indian",4:"Other"}
                    age_bin = age_map.get(age_idx, "")
                    gender  = gen_map.get(gen_idx, "")
                    race    = race_map.get(race_idx, "")

            # 7) save overlay and row
            if args.save_overlays:
                out_img = outdir / (Path(p).stem + "_overlay.jpg")
                overlay.save(str(out_img), quality=92)
                print(f"[overlay] {out_img}")

            # also store raw model angles (for debugging)
            # re-run quickly to fetch exact raw preds
            with torch.no_grad():
                yraw = model(TF_ETH(crop).unsqueeze(0).to(device))[0].detach().cpu().numpy()
            pr_pitch_deg = float(yraw[0]*RAD2DEG); pr_yaw_deg=float(yraw[1]*RAD2DEG)
            pr_pitch_wr = wrap_deg(pr_pitch_deg);   pr_yaw_wr = wrap_deg(pr_yaw_deg)

            row = [p, f"{yaw_ui:.2f}", f"{pitch_ui:.2f}", int(args.mirror_x), int(args.flip_ui_yaw),
                   int(args.flip_ui_pitch), f"{roll_deg:.2f}", f"{pr_pitch_deg:.2f}", f"{pr_yaw_deg:.2f}",
                   f"{pr_pitch_wr:.2f}", f"{pr_yaw_wr:.2f}"]
            if demo_head is not None:
                row += [age_bin, gender, race]
            w.writerow(row)

    print(f"[done] CSV: {csv_path}")
    if args.save_overlays:
        print(f"[done] overlays -> {outdir}")

if __name__ == "__main__":
    main()

