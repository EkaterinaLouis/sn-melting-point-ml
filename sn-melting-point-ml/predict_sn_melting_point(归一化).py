#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import sys
from pathlib import Path
from typing import Dict, Optional

import joblib
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import ttk, messagebox

# 图片显示（建议安装 Pillow）
try:
    from PIL import Image, ImageTk  # pip install pillow
    PIL_OK = True
except Exception:
    PIL_OK = False

ELEMENTS = ['Ag','Al','Au','Bi','Cu','Ga','In','Ni','Pb','Sb','Sn','Zn']

# 放宽公差，先 5e-3，再回退到小数位匹配
TOL_PRIMARY = 5e-3   # 0.005
ROUND_FALLBACK_1 = 4 # 四舍五入到 4 位
ROUND_FALLBACK_2 = 2 # 四舍五入到 2 位（与训练分组一致）

# ---------- 工具 ----------
def project_root() -> Path:
    cwd = Path.cwd()
    markers = ["predict_sn_melting_point.py", "train_and_evaluate_sn_melting_point.py", "sn_melting_gui.py"]
    if any((cwd / m).exists() for m in markers):
        return cwd
    return Path(__file__).resolve().parent

def find_model(root: Path) -> Path:
    best = root / "outputs" / "best_model.joblib"
    if best.exists(): return best
    outputs = root / "outputs"
    if outputs.exists():
        joblibs = sorted(outputs.glob("*.joblib"), key=lambda p: p.stat().st_mtime, reverse=True)
        if joblibs: return joblibs[0]
    messagebox.showerror("模型缺失", "未在 ./outputs/ 下找到模型（*.joblib）。请先运行训练脚本。")
    sys.exit(2)

def find_dataset(root: Path) -> Optional[Path]:
    for cand in [root / "data" / "SnMeltingPoint.xlsx", root / "SnMeltingPoint.xlsx"]:
        if cand.exists(): return cand
    return None

def find_pred_fig(root: Path) -> Optional[Path]:
    for cand in [root / "outputs" / "figures" / "pred_vs_actual.png",
                 root / "outputs" / "pred_vs_actual.png"]:
        if cand.exists(): return cand
    return None

def load_metrics(root: Path) -> Optional[dict]:
    mpath = root / "outputs" / "metrics.json"
    if mpath.exists():
        try:
            return json.loads(mpath.read_text(encoding="utf-8"))
        except Exception:
            return None
    return None

def build_domain_features(X: pd.DataFrame) -> pd.DataFrame:
    x = X[ELEMENTS].values
    x_safe = np.where(x > 0, x, 1.0)
    X["mixing_entropy"] = -np.sum(x * np.log(x_safe), axis=1)
    X["num_components"] = (x > 0).sum(axis=1)
    X["max_frac"] = x.max(axis=1)
    X["min_frac"] = x.min(axis=1)
    X["var_frac"] = x.var(axis=1)
    X["gini_diversity"] = 1.0 - (x**2).sum(axis=1)
    X["sn_frac"] = X["Sn"]
    X["sn_major"] = (X["Sn"] >= 0.5).astype(int)
    return X

# ---------- 更稳健的“已有数据”匹配 ----------
def _mask_tolerance(df: pd.DataFrame, comp: Dict[str, float], tol: float) -> np.ndarray:
    mask = np.ones(len(df), dtype=bool)
    for el in ELEMENTS:
        vals = pd.to_numeric(df[el], errors="coerce").values
        mask &= np.isfinite(vals)
        mask &= np.abs(vals - comp[el]) <= tol
    return mask

def _mask_rounded(df: pd.DataFrame, comp: Dict[str, float], ndigits: int) -> np.ndarray:
    mask = np.ones(len(df), dtype=bool)
    for el in ELEMENTS:
        vals = pd.to_numeric(df[el], errors="coerce").round(ndigits).values
        mask &= np.isfinite(vals)
        mask &= (vals == round(comp[el], ndigits))
    return mask

def lookup_existing_tmelt_robust(df_raw: Optional[pd.DataFrame], comp: Dict[str, float]) -> Optional[float]:
    """三步：公差 -> 四舍五入到4位 -> 四舍五入到2位；返回 LIQUID 最低 T。"""
    if df_raw is None or df_raw.empty:
        return None

    # 必要列
    need_cols = set(ELEMENTS + ["T", "Phase"])
    if not need_cols.issubset(df_raw.columns):
        return None

    # 1) 公差匹配
    mask = _mask_tolerance(df_raw, comp, TOL_PRIMARY)
    cands = df_raw.loc[mask]
    liq = cands[cands["Phase"].astype(str).str.contains("LIQUID", case=False, na=False)]
    if not liq.empty:
        return float(liq["T"].min())

    # 2) 四舍五入到 4 位匹配
    mask4 = _mask_rounded(df_raw, comp, ROUND_FALLBACK_1)
    cands4 = df_raw.loc[mask4]
    liq4 = cands4[cands4["Phase"].astype(str).str.contains("LIQUID", case=False, na=False)]
    if not liq4.empty:
        return float(liq4["T"].min())

    # 3) 四舍五入到 2 位匹配（与训练分组一致）
    mask2 = _mask_rounded(df_raw, comp, ROUND_FALLBACK_2)
    cands2 = df_raw.loc[mask2]
    liq2 = cands2[cands2["Phase"].astype(str).str.contains("LIQUID", case=False, na=False)]
    if not liq2.empty:
        return float(liq2["T"].min())

    return None

# ---------- GUI ----------
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Sn-based Alloy Melting Point（归一化）")
        self.geometry("840x580")
        self.root = project_root()
        self.model_path = find_model(self.root)
        self.dataset_path = find_dataset(self.root)
        self.metrics = load_metrics(self.root)
        self.pipe = joblib.load(self.model_path)
        self.pred_fig_path = find_pred_fig(self.root)
        self.img_ref = None

        # 预读数据表（只读一次，避免每次点击都读盘）
        self.df_raw = None
        if self.dataset_path is not None:
            try:
                self.df_raw = pd.read_excel(self.dataset_path)
            except Exception:
                self.df_raw = None

        main = ttk.Frame(self, padding=10)
        main.pack(fill="both", expand=True)

        # 左侧：输入、结果、评估
        left = ttk.Frame(main)
        left.pack(side="left", fill="y")

        grid = ttk.LabelFrame(left, text="输入各元素分数（可不等于1，程序会归一化）", padding=10)
        grid.pack(fill="x")
        self.entries: Dict[str, tk.Entry] = {}
        for i, el in enumerate(ELEMENTS):
            r, c = divmod(i, 6)
            ttk.Label(grid, text=el).grid(row=r*2, column=c, padx=6, pady=(2,0), sticky="w")
            ent = ttk.Entry(grid, width=8)
            ent.grid(row=r*2+1, column=c, padx=6, pady=(0,6))
            ent.insert(0, "0.0")
            self.entries[el] = ent

        btns = ttk.Frame(left)
        btns.pack(fill="x", pady=6)
        ttk.Button(btns, text="填入示例", command=self.fill_example).pack(side="left", padx=3)
        ttk.Button(btns, text="归一化到和=1", command=self.normalize_inputs).pack(side="left", padx=3)
        ttk.Button(btns, text="预测", command=self.predict).pack(side="left", padx=3)

        res = ttk.LabelFrame(left, text="结果（单位同数据集）", padding=10)
        res.pack(fill="x", pady=(6,0))
        self.pred_var = tk.StringVar(value="—")
        self.gt_var = tk.StringVar(value="（没有数据）")
        ttk.Label(res, text="预测值 Pred:").grid(row=0, column=0, sticky="w")
        ttk.Label(res, textvariable=self.pred_var, font=("Segoe UI", 11, "bold")).grid(row=0, column=1, sticky="w", padx=6)
        ttk.Label(res, text="已有值 Data:").grid(row=1, column=0, sticky="w")
        ttk.Label(res, textvariable=self.gt_var, font=("Segoe UI", 10)).grid(row=1, column=1, sticky="w", padx=6)

        evalf = ttk.LabelFrame(left, text="模型评估（留出测试集）", padding=10)
        evalf.pack(fill="x", pady=(6,0))
        if self.metrics:
            m = self.metrics
            ttk.Label(evalf, text=f"MAE:  {m.get('test_MAE', float('nan')):.2f}").pack(anchor="w")
            ttk.Label(evalf, text=f"RMSE: {m.get('test_RMSE', float('nan')):.2f}").pack(anchor="w")
            ttk.Label(evalf, text=f"R²:   {m.get('test_R2', float('nan')):.4f}").pack(anchor="w")
            ttk.Label(evalf, text=f"n_train: {m.get('n_train','?')}   n_test: {m.get('n_test','?')}   n_features: {m.get('n_features','?')}").pack(anchor="w")
        else:
            ttk.Label(evalf, text="未找到 outputs/metrics.json；下方展示预测-真实散点图。", foreground="#666").pack(anchor="w")

        right = ttk.LabelFrame(main, text="Pred vs Actual（测试集）", padding=10)
        right.pack(side="left", fill="both", expand=True, padx=(10,0))
        self.img_label = ttk.Label(right)
        self.img_label.pack(fill="both", expand=True)
        self.load_and_show_image()

    # ----- 输入/预测 -----
    def fill_example(self):
        example = {"Ag":0.0,"Al":0.0,"Au":0.0,"Bi":0.0,"Cu":0.10,"Ga":0.0,"In":0.10,"Ni":0.0,"Pb":0.0,"Sb":0.0,"Sn":0.70,"Zn":0.10}
        for el, v in example.items():
            self.entries[el].delete(0, tk.END); self.entries[el].insert(0, str(v))

    def normalize_inputs(self) -> Dict[str, float]:
        comp = self.read_inputs(normalize=True)
        messagebox.showinfo("完成", "已将输入归一化到和=1。")
        return comp

    def read_inputs(self, normalize: bool = True) -> Dict[str, float]:
        comp = {}
        for el, ent in self.entries.items():
            txt = ent.get().strip()
            try:
                val = float(txt)
            except Exception:
                val = 0.0
            if val < 0: val = 0.0
            comp[el] = val
        s = sum(comp.values())
        if s <= 0:
            comp = {k:0.0 for k in ELEMENTS}; comp["Sn"]=1.0; s=1.0
        if normalize and not np.isclose(s, 1.0, atol=1e-6):
            comp = {k: v/s for k, v in comp.items()}
        # 回写规范化值（6 位小数）
        for el in ELEMENTS:
            self.entries[el].delete(0, tk.END)
            self.entries[el].insert(0, f"{comp[el]:.6f}")
        return comp

    def predict(self):
        comp = self.read_inputs(normalize=True)

        # 预测
        X = pd.DataFrame([comp], columns=ELEMENTS)
        X = build_domain_features(X)
        pred = float(self.pipe.predict(X)[0])
        self.pred_var.set(f"{pred:.2f}")

        # 稳健查已有数据
        gt = lookup_existing_tmelt_robust(self.df_raw, comp)
        self.gt_var.set("（没有数据）" if gt is None else f"{gt:.2f}")

    # ----- 图像 -----
    def load_and_show_image(self):
        fig_path = find_pred_fig(self.root)
        if fig_path is None:
            self.img_label.config(text="未找到图片：outputs/figures/pred_vs_actual.png")
            return
        wmax, hmax = 520, 520
        try:
            if PIL_OK:
                img = Image.open(fig_path)
                w, h = img.size
                scale = min(wmax / w, hmax / h, 1.0)
                if scale < 1.0:
                    img = img.resize((int(w*scale), int(h*scale)), Image.LANCZOS)
                self.img_ref = ImageTk.PhotoImage(img)
                self.img_label.config(image=self.img_ref)
            else:
                try:
                    img = tk.PhotoImage(file=fig_path.as_posix())
                    self.img_ref = img
                    self.img_label.config(image=self.img_ref)
                except Exception:
                    self.img_label.config(text="无法显示 PNG 图像。请安装 Pillow：pip install pillow")
        except Exception as e:
            self.img_label.config(text=f"加载图像失败：{e}")

if __name__ == "__main__":
    App().mainloop()
