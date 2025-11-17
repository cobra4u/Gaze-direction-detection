import os, argparse, re
import numpy as np, pandas as pd
TO_DEG=180/np.pi
wrap180=lambda a:((a+180)%360)-180
def _sid_from_path(s):
    if not isinstance(s,str): return None
    m=re.search(r"(subject\d{4})",s); return m.group(1) if m else None
def load_preds(p):
    df=pd.read_csv(p)
    if "subject_id" not in df.columns:
        if "subject" in df.columns: df=df.rename(columns={"subject":"subject_id"})
        else: df["subject_id"]=df.get("path","").astype(str).map(_sid_from_path).fillna("unknown")
    for c in ["gt_pitch","gt_yaw","pr_pitch","pr_yaw","subject_id"]:
        if c not in df.columns: raise ValueError(f"Preds CSV missing {c}")
    ep=wrap180((df["pr_pitch"]-df["gt_pitch"])*TO_DEG)
    ey=wrap180((df["pr_yaw"]-df["gt_yaw"])*TO_DEG)
    df["err_pitch_deg"]=ep; df["err_yaw_deg"]=ey; df["err_avg_deg"]=(ep.abs()+ey.abs())/2
    return df
def load_demo(p):
    d=pd.read_csv(p)
    if "subject_id" not in d.columns:
        if "subject" in d.columns: d=d.rename(columns={"subject":"subject_id"})
        elif "sid" in d.columns:   d=d.rename(columns={"sid":"subject_id"})
    if "age_bin" not in d.columns:
        for k in("age_group","agebin","ageBin"):
            if k in d.columns: d=d.rename(columns={k:"age_bin"})
    if "gender" not in d.columns:
        if "sex" in d.columns: d=d.rename(columns={"sex":"gender"})
    need={"subject_id","gender","age_bin"}
    if not need.issubset(d.columns):
        raise ValueError("Demographics must have subject_id, gender, age_bin")
    return d[["subject_id","gender","age_bin"]].copy()
def agg_table(df,key,out_csv):
    g=(df.groupby(key,dropna=False)
         .agg(N_samples=("err_avg_deg","size"),
              MAE_pitch_deg=("err_pitch_deg",lambda s:float(s.abs().mean())),
              MAE_yaw_deg  =("err_yaw_deg",  lambda s:float(s.abs().mean())),
              MAE_avg_deg  =("err_avg_deg","mean")).reset_index())
    subs=(df[["subject_id",key]].drop_duplicates()
            .groupby(key,dropna=False).size().rename("N_subjects").reset_index())
    g=g.merge(subs,on=key,how="left")
    if key=="gender":
        cats=["female","male","unknown"]; g[key]=pd.Categorical(g[key],cats,True); g=g.sort_values(key)
    if key=="age_bin":
        cats=["18-34","35-54","55+","unknown"]; g[key]=pd.Categorical(g[key],cats,True); g=g.sort_values(key)
    os.makedirs(os.path.dirname(out_csv),exist_ok=True); g.to_csv(out_csv,index=False); print(f"[WRITE] {out_csv}"); return g
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--preds_csv",required=True)
    ap.add_argument("--demo_csv", required=True)
    ap.add_argument("--out_dir",  required=True)
    a=ap.parse_args()
    preds=load_preds(a.preds_csv); demo=load_demo(a.demo_csv)
    df=preds.merge(demo,on="subject_id",how="left")
    df["gender"]=df["gender"].fillna("unknown"); df["age_bin"]=df["age_bin"].fillna("unknown")
    agg_table(df,"gender", os.path.join(a.out_dir,"ETH_mae_by_gender.csv"))
    agg_table(df,"age_bin",os.path.join(a.out_dir,"ETH_mae_by_agebin.csv"))
if __name__=="__main__": main()
