#!/usr/bin/env python3
import os, sys, glob, argparse, time, re
from collections import OrderedDict
import numpy as np
import pandas as pd
import h5py
from PIL import Image

import torch
import torch.nn as nn
import torchvision.models as tvm
import torchvision.transforms as T

# ----------------- helpers -----------------
def parse_h5_path(token: str):
    parts = token.split("::")
    if len(parts) != 3:
        return (None,None,None)
    h5_path, dset, frame = parts
    m = re.fullmatch(r"frame(\d+)", frame)
    if not m:
        return (None,None,None)
    return (h5_path, dset, int(m.group(1)))

def ensure_hwc_uint8(arr):
    a = np.asarray(arr)
    if a.ndim == 4: a = a[0]
    if a.ndim == 3 and a.shape[0] == 3: a = np.transpose(a, (1,2,0))
    if a.dtype != np.uint8:
        if a.max() <= 1.0: a = (a*255).clip(0,255).astype(np.uint8)
        else: a = a.clip(0,255).astype(np.uint8)
    return a

# --- KEY FIX: normalize checkpoint key names (DON'T strip 'backbone.') ---
def normalize_backbone_keys(sd: dict):
    """Map m.* or backbone.m.* into backbone.*; leave backbone.* as-is."""
    out = OrderedDict()
    for k, v in sd.items():
        nk = k
        if nk.startswith("backbone.m."):
            nk = "backbone." + nk[len("backbone.m."):]
        elif nk.startswith("m."):
            nk = "backbone." + nk[2:]
        # DO NOT strip 'backbone.'; the model expects it
        out[nk] = v
    return out

# ----------------- dynamic head builder -----------------
class Head1(nn.Module):  # 2048 -> 2
    def __init__(self): super().__init__(); self.fc = nn.Linear(2048,2)
    def forward(self, x): return self.fc(x)

class Head2(nn.Module):  # 2048 -> 512 -> 2
    def __init__(self): super().__init__(); self.fc1 = nn.Linear(2048,512); self.fc2 = nn.Linear(512,2)
    def forward(self,x): return self.fc2(self.fc1(x))

class GazeNet(nn.Module):
    def __init__(self):
        super().__init__()
        m = tvm.resnet50(weights=None)
        m.fc = nn.Identity()
        self.backbone = m
        self.head = Head1()  # will be replaced after checkpoint inspection
    def set_head(self, head_module: nn.Module):
        self.head = head_module
    def forward(self, x):
        f = self.backbone(x)
        return self.head(f)

def find_head_layout(sd: dict):
    # 1-layer direct?
    hw, hb = sd.get("head.weight"), sd.get("head.bias")
    if isinstance(hw, torch.Tensor) and hw.shape == torch.Size([2,2048]):
        return {"layout":"1layer", "names":{"head_w":"head.weight","head_b":"head.bias"}}

    # common 2-layer: shared.0 + gaze_head
    ghw, ghb = sd.get("gaze_head.weight"), sd.get("gaze_head.bias")
    sw0, sb0 = sd.get("shared.0.weight"), sd.get("shared.0.bias")
    if all(isinstance(x, torch.Tensor) for x in [ghw,ghb,sw0,sb0]) and \
       ghw.shape[0]==2 and ghw.shape[1]==sw0.shape[0] and sw0.shape[1]==2048:
        return {"layout":"2layer", "names":{"fc1_w":"shared.0.weight","fc1_b":"shared.0.bias",
                                            "fc2_w":"gaze_head.weight","fc2_b":"gaze_head.bias"}}

    # general scan: (mid,2048) + (2,mid)
    firsts=[]; seconds=[]
    for k,v in sd.items():
        if not isinstance(v, torch.Tensor): continue
        if v.ndim==2 and v.shape[1]==2048 and v.shape[0] in (256,512,1024):
            b = k.rsplit(".",1)[0]+".bias"
            if b in sd and isinstance(sd[b], torch.Tensor) and sd[b].shape==(v.shape[0],):
                firsts.append((k,b,v.shape[0]))
        if v.ndim==2 and v.shape[0]==2 and v.shape[1] in (256,512,1024):
            b = k.rsplit(".",1)[0]+".bias"
            if b in sd and isinstance(sd[b], torch.Tensor) and sd[b].shape==(2,):
                seconds.append((k,b,v.shape[1]))
    for mid in (512,256,1024):
        f = [t for t in firsts if t[2]==mid]
        s = [t for t in seconds if t[2]==mid]
        if f and s:
            fc1_w,fc1_b,_ = f[0]; fc2_w,fc2_b,_ = s[0]
            return {"layout":"2layer","names":{"fc1_w":fc1_w,"fc1_b":fc1_b,"fc2_w":fc2_w,"fc2_b":fc2_b}}

    if isinstance(hw, torch.Tensor) and hw.shape == torch.Size([2,512]):
        return {"layout":"2layer_nofirst", "names":{"fc2_w":"head.weight","fc2_b":"head.bias"}}
    return {"layout":"unknown","names":{}}

def load_eth_state_dict_dynamic(ckpt_path, model: GazeNet):
    blob = torch.load(ckpt_path, map_location="cpu")
    if isinstance(blob, dict):
        if isinstance(blob.get("state_dict"), dict): sd = blob["state_dict"]
        elif "model" in blob:
            sd = blob["model"]
            if hasattr(sd, "state_dict"): sd = sd.state_dict()
        else: sd = blob
    else:
        sd = blob.state_dict() if hasattr(blob, "state_dict") else blob

    # Normalize backbone key space (crucial fix)
    sd = normalize_backbone_keys(sd)

    # Head layout detection & remap
    layout = find_head_layout(sd)
    print("[ckpt] detected layout:", layout["layout"])

    if layout["layout"] == "1layer":
        model.set_head(Head1())
        # ensure names exist; already 'head.*' if found
    elif layout["layout"] == "2layer":
        model.set_head(Head2())
        names = layout["names"]
        # copy to expected names
        sd = OrderedDict(sd)
        sd["head.fc1.weight"] = sd.pop(names["fc1_w"])
        sd["head.fc1.bias"]   = sd.pop(names["fc1_b"])
        sd["head.fc2.weight"] = sd.pop(names["fc2_w"])
        sd["head.fc2.bias"]   = sd.pop(names["fc2_b"])
    elif layout["layout"] == "2layer_nofirst":
        print("[ckpt] WARNING: found only (2,512) head; creating 2-layer head and loading fc2 only.")
        model.set_head(Head2())
        names = layout["names"]
        sd = OrderedDict(sd)
        sd["head.fc2.weight"] = sd.pop(names["fc2_w"])
        sd["head.fc2.bias"]   = sd.pop(names["fc2_b"])
    else:
        print("[ckpt] WARNING: unknown head layout; trying 1-layer.")
        model.set_head(Head1())

    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"[ckpt] loaded: {ckpt_path}")
    print(f"[ckpt] missing={len(missing)} unexpected={len(unexpected)}")
    if missing:   print("  missing (first 10):", list(missing)[:10])
    if unexpected:print("  unexpected (first 10):", list(unexpected)[:10])

# ----------------- main -----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts-dir", required=True, help="CSV parts dir with tokens: /path/subj.h5::DATASET::frameN")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--outcsv", required=True)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--device", choices=["auto","cpu","cuda"], default="auto")
    args = ap.parse_args()

    # Load parts
    part_csvs = sorted(glob.glob(os.path.join(args.parts_dir, "subject*.csv")))
    if not part_csvs:
        raise SystemExit(f"No subject*.csv under {args.parts_dir}")
    dfs = [pd.read_csv(p) for p in part_csvs]
    full = pd.concat(dfs, ignore_index=True)
    print(f"Loaded {len(full)} rows, cols={list(full.columns)}")
    for col in ("path","subject_id","gt_pitch","gt_yaw"):
        if col not in full.columns:
            raise SystemExit(f"Missing column {col}")

    if args.stride > 1:
        full = full.iloc[::args.stride].reset_index(drop=True)

    # Model & weights
    model = GazeNet()
    load_eth_state_dict_dynamic(args.ckpt, model)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    model.to(device).eval()

    tfm = T.Compose([
        T.Resize((224,224)),
        T.ToTensor(),
        T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])

    open_files = {}
    def get_frame_image(tok):
        h5_path, dset, idx = parse_h5_path(tok)
        if not h5_path: return None
        try:
            f = open_files.get(h5_path)
            if f is None:
                if not os.path.exists(h5_path): return None
                f = h5py.File(h5_path, "r"); open_files[h5_path] = f
            if dset not in f: return None
            ds = f[dset]
            if idx < 0 or idx >= ds.shape[0]: return None
            arr = ds[idx]
            arr = ensure_hwc_uint8(arr)
            return Image.fromarray(arr, mode="RGB")
        except Exception:
            return None

    rows=[]; batch_imgs=[]; meta=[]; t0=time.time(); N=len(full)
    with torch.no_grad():
        for i,row in enumerate(full.itertuples(index=False), 1):
            tok = getattr(row, "path")
            img = get_frame_image(tok)
            if img is None: continue
            batch_imgs.append(tfm(img))
            meta.append((tok, getattr(row,"subject_id"),
                        float(getattr(row,"gt_pitch")), float(getattr(row,"gt_yaw"))))
            if len(batch_imgs)==args.batch or i==N:
                x = torch.stack(batch_imgs).to(device, non_blocking=True)
                y = model(x).cpu().numpy()  # radians [pitch,yaw]
                for (tok,sub,gtp,gty), (prp,pry) in zip(meta, y):
                    rows.append({"path": tok, "subject_id": sub,
                                 "gt_pitch": gtp, "gt_yaw": gty,
                                 "pr_pitch": float(prp), "pr_yaw": float(pry)})
                batch_imgs.clear(); meta.clear()
            if i % 2000 == 0:
                print(f"Scanned {i}/{N} in {time.time()-t0:.1f}s, rows={len(rows)}")

    for f in open_files.values():
        try: f.close()
        except: pass

    out = pd.DataFrame(rows)
    out.to_csv(args.outcsv, index=False)
    print(f"Wrote {args.outcsv} rows={len(out)} stride={args.stride} device={device}")

if __name__ == "__main__":
    main()

