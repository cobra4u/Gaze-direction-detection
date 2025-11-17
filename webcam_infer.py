#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, math, time, csv, argparse
from pathlib import Path
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

import torch, torch.nn as nn
import torchvision.transforms as T
import torchvision.models as models

# ---------------- Utils ----------------
def wrap_deg(x: float) -> float:
    return (x + 180.0) % 360.0 - 180.0

def pil_from_bgr(bgr: np.ndarray):
    return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

def bgr_from_pil(pil):
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

def draw_overlay(pil_img: Image.Image, yaw_ui: float, pitch_ui: float, ui_roll_deg: float=0.0):
    img = pil_img.copy().convert("RGB")
    w, h = img.size
    cx, cy = w // 2, h // 2
    axis_len = int(min(w, h) * 0.25)
    dr = ImageDraw.Draw(img)
    # axes
    dr.line([(cx-axis_len,cy),(cx+axis_len,cy)], fill=(255,0,0), width=4)     # +yaw right
    dr.line([(cx,cy+axis_len),(cx,cy-axis_len)], fill=(0,128,255), width=4)   # +pitch up
    # optional visual roll (rotate vector in the plane)
    vy, vp = yaw_ui, pitch_ui
    if abs(ui_roll_deg) > 1e-6:
        r = math.radians(ui_roll_deg)
        vy2 =  vy*math.cos(r) - vp*math.sin(r)
        vp2 =  vy*math.sin(r) + vp*math.cos(r)
        vy, vp = vy2, vp2
    scale = axis_len / 90.0
    dx = vy * scale
    dy = -vp * scale
    ex, ey = int(cx+dx), int(cy+dy)
    dr.line([(cx,cy),(ex,ey)], fill=(255,255,0), width=5)
    ang = math.atan2(ey-cy, ex-cx)
    ah = 16
    left  = (ex - ah*math.cos(ang - math.pi/6), ey - ah*math.sin(ang - math.pi/6))
    right = (ex - ah*math.cos(ang + math.pi/6), ey - ah*math.sin(ang + math.pi/6))
    dr.polygon([(ex,ey), left, right], fill=(255,255,0))
    # label
    try: font = ImageFont.truetype("DejaVuSans.ttf", 22)
    except: font = ImageFont.load_default()
    label = f"UI: yaw={yaw_ui:+.1f}°, pitch={pitch_ui:+.1f}°  (+yaw=right, +pitch=up)"
    ImageDraw.Draw(img).rectangle([10,10,10+len(label)*9,40], fill=(0,0,0,180))
    ImageDraw.Draw(img).text((14,14), label, fill=(255,255,255), font=font)
    return img

# Haar eye detector for roll leveling
def get_cascade(name):
    p = os.path.join(cv2.data.haarcascades, name)
    if not os.path.exists(p):
        raise FileNotFoundError(f"Haar not found: {p}")
    return cv2.CascadeClassifier(p)

EYE_CASCADE = get_cascade("haarcascade_eye.xml")
FACE_CASCADE = get_cascade("haarcascade_frontalface_default.xml")

def auto_level_roll(bgr: np.ndarray, face_hint=None):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    if face_hint is None:
        faces = FACE_CASCADE.detectMultiScale(gray, 1.1, 5, minSize=(120,120))
        if len(faces) == 0: return bgr, 0.0
        faces = sorted(faces, key=lambda r: r[2]*r[3], reverse=True)
        x,y,w,h = faces[0]
    else:
        x,y,w,h = face_hint
    roi = gray[y:y+h, x:x+w]
    eyes = EYE_CASCADE.detectMultiScale(roi, 1.1, 5, minSize=(30,30))
    if len(eyes) < 2: return bgr, 0.0
    eyes = sorted(eyes, key=lambda e: e[0])[:2]
    c = []
    for (ex,ey,ew,eh) in eyes:
        c.append((x+ex+ew/2, y+ey+eh/2))
    (x1,y1),(x2,y2) = c[0], c[1]
    angle = math.degrees(math.atan2((y2-y1),(x2-x1)))
    # rotate image by negative angle to level eyes
    h2,w2 = bgr.shape[:2]
    M = cv2.getRotationMatrix2D((w2/2,h2/2), angle, 1.0)
    rot = cv2.warpAffine(bgr, M, (w2,h2), flags=cv2.INTER_LINEAR)
    return rot, angle

# ------------- Model -------------
class GazeResNet(nn.Module):
    def __init__(self, backbone="resnet50"):
        super().__init__()
        if backbone == "resnet18":
            m = models.resnet18(weights=None); feat=512
        elif backbone == "resnet50":
            m = models.resnet50(weights=None); feat=2048
        else:
            raise ValueError("backbone must be resnet18 or resnet50")
        m.fc = nn.Identity()
        self.base = m
        self.head = nn.Linear(feat, 2)
    def forward(self, x): return self.head(self.base(x))

TF = T.Compose([
    T.Resize((224,224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
])

# ------------- Main -------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cam", type=int, default=0, help="OpenCV camera index")
    ap.add_argument("--ckpt", default="/users/project1/pt01281/gaze_outputs/checkpoints_xgaze50/gaze_best.pth")
    ap.add_argument("--backbone", choices=["resnet18","resnet50"], default="resnet50")
    ap.add_argument("--outdir", default="/users/kdm/divjots2002/gazemain/my_gaze_outputs")
    ap.add_argument("--csv", default=None)
    ap.add_argument("--default-mirror-x", action="store_true", help="Start with mirroring ON (laptop selfie)")
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    csv_path = Path(args.csv) if args.csv else (outdir / "_webcam_log.csv")
    write_header = not csv_path.exists()

    # model
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = GazeResNet(args.backbone).to(dev).eval()
    ckpt = torch.load(args.ckpt, map_location="cpu")
    if "state_dict" in ckpt: ckpt = ckpt["state_dict"]
    model.load_state_dict(ckpt, strict=False)
    rad2deg = 180.0 / math.pi

    # session toggles (stateless on disk)
    mirror_x   = args.default_mirror_x
    flip_yaw   = False
    flip_pitch = True   # matches your reliable setting
    ui_roll    = 0.0    # extra visual roll if needed

    cap = cv2.VideoCapture(args.cam)
    if not cap.isOpened():
        print("[err] cannot open camera"); return

    print("[keys] SPACE=save  m=mirror  y=flip yaw  p=flip pitch  r/R=roll ±2°  q=quit")
    while True:
        ok, frame = cap.read()
        if not ok: break

        # mirror preview if chosen
        vis = cv2.flip(frame, 1) if mirror_x else frame

        # auto-level head roll using eyes (on the shown frame)
        leveled, est_roll = auto_level_roll(vis)
        pil = pil_from_bgr(leveled)

        # inference (single frame, stateless)
        with torch.no_grad():
            x = TF(pil).unsqueeze(0).to(dev)
            y = model(x)[0].detach().cpu().numpy()
        pr_pitch_deg = wrap_deg(float(y[0])*rad2deg)
        pr_yaw_deg   = wrap_deg(float(y[1])*rad2deg)

        # camera->UI (+yaw right, +pitch up)
        yaw_ui   = -pr_yaw_deg
        pitch_ui = -pr_pitch_deg
        # if we mirrored the image, horizontal is already reversed visually -> flip back
        if mirror_x:
            yaw_ui = -yaw_ui
        if flip_yaw:
            yaw_ui = -yaw_ui
        if flip_pitch:
            pitch_ui = -pitch_ui

        # draw overlay (optionally add small ui_roll)
        overlay_pil = draw_overlay(pil, yaw_ui, pitch_ui, ui_roll_deg=ui_roll)
        overlay_bgr = bgr_from_pil(overlay_pil)

        # heads-up text
        hud = f"mirror:{int(mirror_x)} yawFlip:{int(flip_yaw)} pitchFlip:{int(flip_pitch)} autoRoll:{est_roll:+.1f} uiRoll:{ui_roll:+.1f}"
        cv2.putText(overlay_bgr, hud, (10, overlay_bgr.shape[0]-15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20,220,20), 2)

        cv2.imshow("webcam-gaze (SPACE=save, m/y/p/r/R/q)", overlay_bgr)
        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'):
            break
        elif k == ord('m'):
            mirror_x = not mirror_x
        elif k == ord('y'):
            flip_yaw = not flip_yaw
        elif k == ord('p'):
            flip_pitch = not flip_pitch
        elif k == ord('r'):
            ui_roll -= 2.0
        elif k == ord('R'):
            ui_roll += 2.0
        elif k == 32:  # SPACE
            ts = time.strftime("%Y%m%d-%H%M%S")
            out_img = outdir / f"webcam_{ts}_overlay.jpg"
            cv2.imwrite(str(out_img), overlay_bgr)
            with open(csv_path, "a", newline="") as f:
                w = csv.writer(f)
                if write_header:
                    w.writerow(["path","pr_pitch_deg","pr_yaw_deg","ui_yaw_deg","ui_pitch_deg",
                                "mirror_x","flip_yaw","flip_pitch","auto_roll_deg","ui_roll_deg"])
                    write_header=False
                w.writerow([str(out_img), f"{pr_pitch_deg:.2f}", f"{pr_yaw_deg:.2f}",
                            f"{yaw_ui:.2f}", f"{pitch_ui:.2f}",
                            int(mirror_x), int(flip_yaw), int(flip_pitch),
                            f"{est_roll:.1f}", f"{ui_roll:.1f}"])
            print(f"[saved] {out_img}")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

