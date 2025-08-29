#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PyQt6 GUI: Pick DICOM folder or NIfTI → pick target ROI(s) → run TotalSegmentator
→ export STL mesh (mm) and per-slice SVG masks (optional), plus volume CSV (optional).

Dependencies:
  pip install SimpleITK PyQt6 totalsegmentator nibabel numpy scikit-image trimesh svgwrite

Notes:
- Requires TotalSegmentator CLI available in PATH (TotalSegmentator / TotalSegmentator.exe)
- Default output folder: <parent>/<input_name>_Instant3D
    e.g., C:/data/CTseries  →  C:/data/CTseries_Instant3D
          C:/data/scan001.nii.gz  →  C:/data/scan001_Instant3D
- NIfTI masks (.nii/.nii.gz):
    Keep the original masks exactly as output by TotalSegmentator (e.g., liver.nii.gz).
    Do NOT re-save masks with input-name prefixes.
- STL / SVG / CSV naming:
    Use "<input_name>_<roi>_mesh.stl", "svg_<input_name>_<roi>/mask0001.svg...", "volume_<input_name>_<roi>.csv".
- ROI aliases combine multiple masks into one mesh/SVG (e.g., "pelvis").
- All status strings are English.
"""


from __future__ import annotations
import os
import sys
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import nibabel as nib
from skimage import measure
import trimesh

from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QApplication, QWidget, QFileDialog, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QComboBox, QCheckBox, QSpinBox,
    QTextEdit, QProgressBar, QGroupBox, QFormLayout, QMessageBox
)

from PyQt6.QtGui import QTextCursor

import csv
from PIL import Image

import re

from PyQt6.QtWidgets import (QLineEdit, QCompleter, QListWidget, QListWidgetItem, QPushButton)
from PyQt6.QtCore import QStringListModel   # ← QtCore からインポート

from PyQt6.QtGui import QShortcut, QKeySequence

from PyQt6.QtWidgets import QLabel

from PyQt6.QtCore import Qt

try:
    from rapidfuzz import process as rf_process
    _HAS_RAPIDFUZZ = True
except Exception:
    _HAS_RAPIDFUZZ = False
    
from PyQt6.QtWidgets import QInputDialog, QLineEdit, QMessageBox
import json, os
from pathlib import Path

import difflib, os

APP_ROOT = Path(__file__).parent
RES_DIR = APP_ROOT / "resources"






import json

# 2) Task catalogs  -----------------------------------------------------------
# Keep this light-weight. You can extend via resources/roi_catalog_<task>.txt.
# For tasks with many classes (e.g. total), we fall back to your existing
# VALID_ROIS_* sets or to an external txt file if you place one in resources/.

OPEN_TASKS = [
    "total", "total_mr",
    "craniofacial_structures",
    "head_glands_cavities", "head_muscles", "headneck_bones_vessels",
    "headneck_muscles", "oculomotor_muscles",
    "body", "body_mr",
    "lung_vessels", "lung_nodules", "pleural_pericard_effusion",
    "liver_vessels", "liver_segments", "liver_segments_mr",
    "kidney_cysts", "breasts", "abdominal_muscles",
    "vertebrae_mr", "hip_implant", "teeth",
]

LICENSED_TASKS = [
    "heartchambers_highres", "appendicular_bones", "appendicular_bones_mr",
    "tissue_types", "tissue_types_mr", "tissue_4_types",
    "brain_structures", "vertebrae_body",
    "face", "face_mr",
    "thigh_shoulder_muscles", "thigh_shoulder_muscles_mr",
    "coronary_arteries",
]

# Minimal built-in ROI sets for selected subtasks (extend as needed or drop a
# resources/roi_catalog_<task>.txt to override).
TASK_ROIS_MIN = {
    # craniofacial subset
    "craniofacial_structures": {
        "mandible", "teeth_lower", "teeth_upper", "skull", "head",
        "sinus_maxillary", "sinus_frontal",
    },
    "head_glands_cavities": {
        "eye_left", "eye_right", "eye_lens_left", "eye_lens_right",
        "optic_nerve_left", "optic_nerve_right",
        "parotid_gland_left", "parotid_gland_right",
        "submandibular_gland_right", "submandibular_gland_left",
        "nasopharynx", "oropharynx", "hypopharynx",
        "nasal_cavity_right", "nasal_cavity_left",
        "auditory_canal_right", "auditory_canal_left",
        "soft_palate", "hard_palate",
    },
    "head_muscles": {
        "masseter_right", "masseter_left", "temporalis_right", "temporalis_left",
        "lateral_pterygoid_right", "lateral_pterygoid_left",
        "medial_pterygoid_right", "medial_pterygoid_left",
        "tongue", "digastric_right", "digastric_left",
    },
    "headneck_bones_vessels": {
        "larynx_air", "thyroid_cartilage", "hyoid", "cricoid_cartilage",
        "zygomatic_arch_right", "zygomatic_arch_left",
        "styloid_process_right", "styloid_process_left",
        "internal_carotid_artery_right", "internal_carotid_artery_left",
        "internal_jugular_vein_right", "internal_jugular_vein_left",
    },
    "headneck_muscles": {
        "sternocleidomastoid_right", "sternocleidomastoid_left",
        "superior_pharyngeal_constrictor", "middle_pharyngeal_constrictor",
        "inferior_pharyngeal_constrictor",
        "trapezius_right", "trapezius_left", "platysma_right", "platysma_left",
        "levator_scapulae_right", "levator_scapulae_left",
        "anterior_scalene_right", "anterior_scalene_left",
        "middle_scalene_right", "middle_scalene_left",
        "posterior_scalene_right", "posterior_scalene_left",
        "sterno_thyroid_right", "sterno_thyroid_left",
        "thyrohyoid_right", "thyrohyoid_left",
        "prevertebral_right", "prevertebral_left",
    },
    "oculomotor_muscles": {
        "skull", "eyeball_right", "lateral_rectus_muscle_right",
        "superior_oblique_muscle_right", "levator_palpebrae_superioris_right",
        "superior_rectus_muscle_right", "medial_rectus_muscle_left",
        "inferior_oblique_muscle_right", "inferior_rectus_muscle_right",
        "optic_nerve_left", "eyeball_left", "lateral_rectus_muscle_left",
        "superior_oblique_muscle_left", "levator_palpebrae_superioris_left",
        "superior_rectus_muscle_left", "medial_rectus_muscle_right",
        "inferior_oblique_muscle_left", "inferior_rectus_muscle_left",
        "optic_nerve_right",
    },
    "liver_vessels": {"liver_vessels", "liver_tumor"},
    "liver_segments": {
        "liver_segment_1", "liver_segment_2", "liver_segment_3", "liver_segment_4",
        "liver_segment_5", "liver_segment_6", "liver_segment_7", "liver_segment_8",
    },
    "liver_segments_mr": {
        "liver_segment_1", "liver_segment_2", "liver_segment_3", "liver_segment_4",
        "liver_segment_5", "liver_segment_6", "liver_segment_7", "liver_segment_8",
    },
    "lung_vessels": {"lung_vessels", "lung_trachea_bronchia"},
    "pleural_pericard_effusion": {"pleural_effusion", "pericardial_effusion"},
    "lung_nodules": {"lung", "lung_nodules"},
    "kidney_cysts": {"kidney_cyst_left", "kidney_cyst_right"},
    "abdominal_muscles": {
        "pectoralis_major_right", "pectoralis_major_left",
        "rectus_abdominis_right", "rectus_abdominis_left",
        "serratus_anterior_right", "serratus_anterior_left",
        "latissimus_dorsi_right", "latissimus_dorsi_left",
        "trapezius_right", "trapezius_left",
        "external_oblique_right", "external_oblique_left",
        "internal_oblique_right", "internal_oblique_left",
        "erector_spinae_right", "erector_spinae_left",
        "transversospinalis_right", "transversospinalis_left",
        "psoas_major_right", "psoas_major_left",
        "quadratus_lumborum_right", "quadratus_lumborum_left",
    },
    "vertebrae_mr": {
        "sacrum", "vertebrae_L5", "vertebrae_L4", "vertebrae_L3", "vertebrae_L2", "vertebrae_L1",
        "vertebrae_T12", "vertebrae_T11", "vertebrae_T10", "vertebrae_T9", "vertebrae_T8", "vertebrae_T7",
        "vertebrae_T6", "vertebrae_T5", "vertebrae_T4", "vertebrae_T3", "vertebrae_T2", "vertebrae_T1",
        "vertebrae_C7", "vertebrae_C6", "vertebrae_C5", "vertebrae_C4", "vertebrae_C3", "vertebrae_C2", "vertebrae_C1",
    },
    "breasts": {"breast"},
    "hip_implant": {"hip_implant"},
    # teeth: extremely long; keep empty so it shows no suggestions
    "teeth": set(),
}

# Optional synonyms (UI → CLI)
ROI_ALIASES = {
    "lower_jaw": "mandible",
    "upper_jaw": "maxilla",
}

# 3) Helpers to load per-task ROI catalogs  -----------------------------------
# Try resources/roi_catalog_<task>.txt first. If not present, use TASK_ROIS_MIN.

def load_roi_catalog_for_task(task: str, modality_fallback: str | None = None) -> list[str]:
    task = (task or "").strip()
    # resources
    if task:
        path = RES_DIR / f"roi_catalog_{task}.txt"
        ext = read_roi_file(path)
        if ext:
            return sorted(set(ext))
    # built-in
    if task in TASK_ROIS_MIN:
        return sorted(TASK_ROIS_MIN[task])
    # legacy fallbacks by modality
    if modality_fallback:
        return load_roi_catalog(modality_fallback)
    return []










def read_roi_file(path: Path) -> list[str] | None:
    try:
        if path.exists():
            return [line.strip() for line in path.read_text(encoding="utf-8").splitlines()
                    if line.strip() and not line.strip().startswith("#")]
    except Exception:
        pass
    return None

# def load_roi_catalog(modality: str) -> list[str]:
#     modality = modality.upper()
#     if modality == "MRI":
#         ext = read_roi_file(RES_DIR / "roi_catalog_mr.txt")
#         return sorted(set(ext)) if ext else sorted(VALID_ROIS_MR)
#     if modality == "HEAD":
#         ext = read_roi_file(RES_DIR / "roi_catalog_head.txt")
#         return sorted(set(ext)) if ext else sorted(VALID_ROIS_HEAD)
#     # CT (default)
#     ext = read_roi_file(RES_DIR / "roi_catalog_ct.txt")
#     return sorted(set(ext)) if ext else sorted(VALID_ROIS_CT)
    
def load_roi_catalog(modality: str) -> list[str]:
    m = (modality or "").upper()
    if m == "CT":
        return VALID_ROIS_CT
    if m == "MRI":
        return VALID_ROIS_MR
    return []  # Head はもう扱わない

def fuzzy_topk(query: str, candidates: list[str], k: int = 30) -> list[str]:
    candidates = list(candidates)  # ← これを先頭に追加（set/tuple/iterable でもOKに）
    if not query:
        return candidates[:k]
    try:
        from rapidfuzz import process as rf_process
        res = rf_process.extract(query, candidates, limit=k, score_cutoff=30)
        return [name for name, score, _ in res]
    except Exception:
        part = [c for c in candidates if query.lower() in c.lower()]
        if part:
            return part[:k]
        return difflib.get_close_matches(query, candidates, n=k, cutoff=0.3)
    return candidates[:k]







def _ts_home_dir() -> Path:
    # TOTALSEG_HOME_DIR があれば優先、無ければユーザーのホーム配下
    d = os.environ.get("TOTALSEG_HOME_DIR")
    return Path(d) if d else (Path.home() / ".totalsegmentator")

def has_totalseg_license() -> bool:
    cfg = _ts_home_dir() / "config.json"
    try:
        data = json.loads(cfg.read_text(encoding="utf-8"))
        key = (data.get("license_key") or "").strip()
        return bool(key)
    except Exception:
        return False




# 統一：同義語＆グループを両方適用
ROI_SYNONYM = { "lower_jaw": "mandible", "upper_jaw": "maxilla" }  # 1→1
ROI_GROUPS  = { "pelvis": [...], "kidneys": [...], "lungs": [...] } # 1→N


# ---- ROI definitions -------------------------------------------------
VALID_ROIS_CT = {
    "spleen","liver","pancreas","stomach","gallbladder",
    "kidney_left","kidney_right","adrenal_gland_left","adrenal_gland_right",
    "heart","aorta","inferior_vena_cava","superior_vena_cava",
    "esophagus","trachea","thyroid_gland","small_bowel","duodenum","colon",
    "urinary_bladder","prostate",
    "sacrum","hip_left","hip_right","femur_left","femur_right","skull","brain",
    # ※ 肺は葉ごとなので、left/right は total には無い
    "lung_upper_lobe_left","lung_lower_lobe_left",
    "lung_upper_lobe_right","lung_middle_lobe_right","lung_lower_lobe_right",
}

VALID_ROIS_MR = {
    "brain","heart","liver","spleen","pancreas","stomach","gallbladder",
    "kidney_left","kidney_right","adrenal_gland_left","adrenal_gland_right",
    "urinary_bladder","prostate","uterus","ovary_left","ovary_right",
    "lung_left","lung_right",  # MR は whole lung がある想定（あなたのMRカタログにも記載） :contentReference[oaicite:2]{index=2}
}

VALID_ROIS_HEAD = {
    "mandible","teeth_lower","teeth_upper","skull","head",
    "sinus_maxillary","sinus_frontal",
}

ROI_ALIAS = {
    "pelvis": ["sacrum", "hip_left", "hip_right", "urinary_bladder", "prostate", "uterus"],
    "kidneys": ["kidney_left", "kidney_right"],
    "lungs": ["lung_left", "lung_right"],
}




def get_valid_rois_for(modality: str) -> list[str]:
    if modality.upper() == "MRI":
        return sorted(VALID_ROIS_MR)
    return sorted(VALID_ROIS_CT)




# ----------------- helpers -----------------






def _resolve_rois(modality: str, roi_name: str) -> list[str]:
    # エイリアス → 実体（複数）
    if roi_name in ROI_ALIAS:
        targets = ROI_ALIAS[roi_name]
    else:
        targets = [roi_name]

    valid = set(get_valid_rois_for(modality))
    unsupported = [r for r in targets if r not in valid]
    if unsupported:
        # MRIで未対応などの場合に警告
        QMessageBox.warning(None, "ROI not supported",
                            f"{', '.join(unsupported)} is not available for {modality}.")
        # サポートされるものだけに絞る
        targets = [r for r in targets if r in valid]
    return targets


# def which_totalseg() -> str:
#     exe = shutil.which("TotalSegmentator") or shutil.which("TotalSegmentator.exe")
#     if not exe:
#         raise FileNotFoundError("TotalSegmentator CLI not found in PATH")
#     return exe

def which_totalseg() -> str:
    # 0) 環境変数で明示指定があれば最優先
    env = os.environ.get("TOTALSEGMENTATOR_EXE")
    if env and Path(env).exists():
        return env

    # 1) PATH から探す
    exe = shutil.which("TotalSegmentator") or shutil.which("TotalSegmentator.exe")
    if exe:
        return exe

    # 2) 親切メッセージ
    raise FileNotFoundError(
        "TotalSegmentator not found.\n"
        "Install it with:\n"
        "  pip install totalsegmentator\n"
        "If not on PATH, set env TOTALSEGMENTATOR_EXE to the full .exe path."
    )







def ensure_nifti_input(src_path: Path, tmp_dir: Path) -> Path:
    """
    Return a path to a NIfTI (.nii.gz) that TotalSegmentator can read.
    If src is DICOM folder -> return folder as-is.
    If src is .nii/.nii.gz -> return file as-is.
    If src is .nrrd -> convert to .nii.gz in tmp_dir and return its path.
    """
    if src_path.is_dir():
        return src_path
    ext = src_path.suffix.lower()
    if ext in [".nii", ".gz"] or src_path.name.endswith(".nii.gz"):
        return src_path
    if ext == ".nrrd":
        tmp_dir.mkdir(parents=True, exist_ok=True)
        out_nii = tmp_dir / "input_from_nrrd.nii.gz"
        try:
            import SimpleITK as sitk
            img = sitk.ReadImage(str(src_path))
            sitk.WriteImage(img, str(out_nii))
            return out_nii
        except Exception:
            import nrrd, nibabel as nib, numpy as np
            data, hdr = nrrd.read(str(src_path))
            dirs = hdr.get("space directions")
            origin = hdr.get("space origin", [0, 0, 0])
            aff = np.eye(4)
            try:
                M = np.array([[d[0], d[1], d[2]] for d in dirs], dtype=float)
                aff[:3, :3] = M
            except Exception:
                pass
            try:
                aff[:3, 3] = np.array(origin, dtype=float)
            except Exception:
                pass
            img = nib.Nifti1Image(np.asarray(data), aff)
            nib.save(img, str(out_nii))
            return out_nii
    raise ValueError(f"Unsupported input type: {src_path}")




# ----------------- heavy work in thread -----------------

@dataclass
class Job:
    input_path: Path
    output_dir: Path
    roi_name: str
    plane: str
    reverse_slices: bool
    flip_lr: bool
    flip_ud: bool
    use_cpu: bool
    fastest: bool
    smooth_iters: int
    modality: str          # ← 追加 ("CT" or "MRI")
    task: str = "__auto__"
    robust_crop: bool = False
    export_svg: bool = False
    export_csv: bool = False


class Worker(QThread):
    log = pyqtSignal(str)
    progress = pyqtSignal(int)
    failed = pyqtSignal(str)
    # finished = pyqtSignal(Path, Path, Path)  # (mask_nii, stl_path, svg_dir)
    finished = pyqtSignal(object, object, object)  # svg_dir は None になる場合あり
        
    def __init__(self, job: Job, selected_rois: list[str] | None = None, parent=None):
        super().__init__(parent)
        self.job = job
        self.selected_rois = (selected_rois or []).copy()





                    
    def run_totalseg(self, inp: Path, out_dir: Path, rois: list[str], use_cpu: bool, fastest: bool) -> dict[str, Path]:
        exe = which_totalseg()
        userTask = getattr(self.job, "task", "__auto__") or "__auto__"
    
        if userTask != "__auto__":
            task = userTask
        else:
            modality = (self.job.modality or "CT").upper()
            task = "total" if modality == "CT" else "total_mr"
    
            # （任意）ROIが全て頭頸部セットなら自動で craniofacial に昇格
            cranio = TASK_ROIS_MIN.get("craniofacial_structures", set())
            if rois and all(r.lower() in cranio for r in rois):
                task = "craniofacial_structures"
                
                
    
        # Validate ROIs for the chosen task
        valid = set(load_roi_catalog_for_task(task, self.job.modality))
        bad = [r for r in rois if r not in valid]
        rois = [ROI_ALIASES.get(r, r) for r in rois if r in valid]
        if bad:
            self.log.emit(f"[W] ROI not available for task '{task}': {bad}. Ignored.")
    
    
    
    
            
        # Build args
        args = [exe, "-i", str(inp), "-o", str(out_dir), "-d", ("cpu" if use_cpu else "gpu"), "--task", task]
        
        # v2 uses --fast (3mm). Keep UI label as Fast.
        if fastest:
            args += ["--fast"]
        
        # roi_subset は total / total_mr のみ有効
        allow_subset = task in ("total", "total_mr")
        
        if rois:
            if allow_subset:
                args += ["--roi_subset"] + [r.lower() for r in rois]
                if getattr(self.job, "robust_crop", False):
                    args += ["--robust_crop"]
            else:
                # craniofacial_structures などでは subset を CLI に渡さない
                # （実行後に出力フォルダから post-filter で r を拾う）
                self.log.emit(
                    f"[I] Task '{task}' does not support --roi_subset. "
                    f"Running full task and post-filtering to: {rois}"
                )
        
        self.log.emit("[CMD] " + " ".join(args))
                
        
        
        
                
        home_dir = os.environ.get("TOTALSEG_HOME_DIR", str(Path.home() / ".totalsegmentator"))
        self.log.emit(f"[I] TS home: {home_dir}  (expect config.json with license_key)")
        
        # ← ここを env 直参照ではなく、上で決めた home_dir を使う
        cfg = Path(home_dir) / "config.json"
        try:
            import json
            if cfg.exists():
                data = json.loads(cfg.read_text(encoding="utf-8"))
                k = (data.get("license_key", "")[:8] + "…") if data.get("license_key") else ""
                self.log.emit(f"[I] Found license at {cfg}" + (f" (key starts with: {k})" if k else ""))
            else:
                self.log.emit(f"[W] No license file at {cfg}")
        except Exception as e:
            self.log.emit(f"[W] Could not read license at {cfg}: {e}")

        
        
        
        
        
        
        res = subprocess.run(args, capture_output=True, text=True)
        if res.stdout:
            self.log.emit(res.stdout)
        if res.returncode != 0:
            if res.stderr:
                self.log.emit(res.stderr)
            raise RuntimeError("TotalSegmentator failed")
    
    
    
    
        seg_dir = out_dir / "segmentations"
        if not seg_dir.exists():
            seg_dir = out_dir
    
        nii_files = list(seg_dir.rglob("*.nii")) + list(seg_dir.rglob("*.nii.gz"))
        def find_mask(name: str) -> Path | None:
            name_l = name.lower()
            cand = [p for p in nii_files if name_l in p.name.lower()]
            return cand[0] if cand else None
    
        roi_map: dict[str, Path] = {}
        targets = rois or []
        if not targets:
            # No subset -> return everything
            for p in nii_files:
                key = p.stem.lower().replace(".nii", "")
                roi_map[key] = p
            return roi_map
    
        for r in targets:
            p = find_mask(r)
            if p is None:
                raise FileNotFoundError(f"No mask for ROI '{r}' in {out_dir} (task={task})")
            roi_map[r] = p
        return roi_map    
    
    
    
    
    
    
    

    # def union_masks(self, nii_paths: List[Path]) -> Tuple[Path, nib.Nifti1Image]:
    #     self.log.emit("[I] Loading & uniting masks...")
    #     base_img = nib.load(str(nii_paths[0]))
    #     base_img = nib.as_closest_canonical(base_img)
    #     base_arr = (np.asarray(base_img.get_fdata()) > 0.5).astype(np.uint8)
    #     for p in nii_paths[1:]:
    #         img = nib.load(str(p))
    #         img = nib.as_closest_canonical(img)
    #         arr = (np.asarray(img.get_fdata()) > 0.5).astype(np.uint8)
    #         if arr.shape != base_arr.shape:
    #             raise RuntimeError("Mask shapes do not match between ROIs; cannot union.")
    #         base_arr |= arr
    #     union_img = nib.Nifti1Image(base_arr.astype(np.uint8), base_img.affine, base_img.header)
    #     out_mask = self.job.output_dir / "roi_mask.nii.gz"
    #     nib.save(union_img, str(out_mask))
    #     return out_mask, union_img
    
    # def union_masks(self, nii_paths: List[Path]) -> Tuple[Path, nib.Nifti1Image]:
    #     self.log.emit("[I] Loading & uniting masks...")
    #     base_img = nib.load(str(nii_paths[0]))
    #     base_img = nib.as_closest_canonical(base_img)
    #     base_arr = (np.asarray(base_img.get_fdata()) > 0.5).astype(np.uint8)
    #     for p in nii_paths[1:]:
    #         img = nib.load(str(p))
    #         img = nib.as_closest_canonical(img)
    #         arr = (np.asarray(img.get_fdata()) > 0.5).astype(np.uint8)
    #         if arr.shape != base_arr.shape:
    #             raise RuntimeError("Mask shapes do not match between ROIs; cannot union.")
    #         base_arr |= arr
    
    #     union_img = nib.Nifti1Image(base_arr.astype(np.uint8), base_img.affine, base_img.header)
    
    #     # 入力ベース名を取得 ← ★追加
    #     inp_src = Path(self.job.input_path)
    #     input_base = inp_src.stem if inp_src.is_file() else inp_src.name
    
    #     # ファイル名に input_base を付与 ← ★変更
    #     out_mask = self.job.output_dir / f"{input_base}_roi_mask.nii.gz"
    #     nib.save(union_img, str(out_mask))
    #     return out_mask, union_img
        
    def union_masks(self, nii_paths: List[Path]) -> Tuple[Path, nib.Nifti1Image]:
        self.log.emit("[I] Loading & uniting masks.")
        base_img = nib.load(str(nii_paths[0]))
        base_img = nib.as_closest_canonical(base_img)
        base_arr = (np.asarray(base_img.get_fdata()) > 0.5).astype(np.uint8)
        for p in nii_paths[1:]:
            img = nib.load(str(p))
            img = nib.as_closest_canonical(img)
            arr = (np.asarray(img.get_fdata()) > 0.5).astype(np.uint8)
            if arr.shape != base_arr.shape:
                raise RuntimeError("Mask shapes do not match between ROIs; cannot union.")
            base_arr |= arr
    
        union_img = nib.Nifti1Image(base_arr.astype(np.uint8), base_img.affine, base_img.header)
    
        # ★ 追加：入力ベース名
        inp_src = Path(self.job.input_path)
        input_base = inp_src.stem if inp_src.is_file() else inp_src.name
    
        # ★ 変更：ファイル名に input_base を付与
        out_mask = self.job.output_dir / f"{input_base}_roi_mask.nii.gz"
        nib.save(union_img, str(out_mask))
        return out_mask, union_img
        

            
    def mask_to_stl(self, nifti_img: nib.Nifti1Image, stl_path: Path, smooth_iters: int):
        self.log.emit("[I] Creating STL via marching cubes...")
        data = (np.asarray(nifti_img.get_fdata()) > 0.5).astype(np.uint8)
        sx, sy, sz = nifti_img.header.get_zooms()[:3]
    
        verts, faces, _norms, _vals = measure.marching_cubes(
            data, level=0.5, spacing=(sx, sy, sz)
        )
        mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    
        # --- ▼ 裏表チェック & 修正 ▼ ---
        # outward なら体積は正、inside-out なら負になる
        try:
            if mesh.volume < 0:
                mesh.invert()  # 三角形の巻き方向と法線を反転
        except Exception:
            # volume が計算できない環境用フォールバック
            mesh.faces = mesh.faces[:, ::-1]
        mesh.fix_normals()
        # --- ▲ ここまで ▼ ---
    
        if smooth_iters and smooth_iters > 0:
            trimesh.smoothing.filter_laplacian(mesh, lamb=0.5, iterations=int(smooth_iters))
    
        stl_path.parent.mkdir(parents=True, exist_ok=True)
        mesh.export(str(stl_path))
        self.log.emit(f"[OK] STL saved: {stl_path}")
        
        
        

    
    def mask_to_svgs(self, nifti_img, svg_dir,
                     plane, flip_lr, flip_ud, reverse_slices):
        import svgwrite
        from skimage import measure
        import numpy as np
    
        vol = (np.asarray(nifti_img.get_fdata()) > 0.5).astype(np.uint8)
    
        ax = {"axial": 2, "coronal": 1, "sagittal": 0}.get(plane, 2)
        n = vol.shape[ax]
        order = range(n-1, -1, -1) if reverse_slices else range(n)
    
        sample = np.take(vol, 0, axis=ax)
        h, w = sample.shape
        svg_dir.mkdir(parents=True, exist_ok=True)
    
        # 塗り/線（Obj1 検出のための赤ストロークは維持）
        FILL_COLOR = "#ff0000"
        FILL_OPACITY = 0.7
        STROKE_COLOR = "#ff0000"
        STROKE_WIDTH = 0.01  # 非スケーリング指定を付けるので視覚的にも極細のまま
    
        # ▼ ビューボックスに余白（はみ出し裁ち落とし防止）
        MARGIN = 1.0  # px 単位
    
        for i, k in enumerate(order, start=1):
            plane2d = np.take(vol, k, axis=ax)
            if flip_lr:
                plane2d = np.flip(plane2d, axis=1)
            if flip_ud:
                plane2d = np.flip(plane2d, axis=0)
    
            # 端で途切れないように 1px パディングしてから輪郭
            padded = np.pad(plane2d, 1, mode="constant")
            contours = measure.find_contours(padded, level=0.5)
    
            d_parts = []
            for c in contours:
                # パディングを戻す
                c = c - 1.0
                if len(c) < 3:
                    continue
                # c は (y, x)
                d = "M {:.2f},{:.2f} ".format(c[0,0], c[0,1]) + \
                    " ".join("L {:.2f},{:.2f}".format(y, x) for y, x in c[1:]) + " Z"
                d_parts.append(d)
    
    
                        
            # …略…
            canvas_w, canvas_h = h, w  # ← (y,x) で描くのでキャンバスは (h,w)
            dwg = svgwrite.Drawing(size=("100%", "100%"))  # 絶対サイズなし
            dwg.viewbox(-MARGIN, -MARGIN, canvas_w + 2*MARGIN, canvas_h + 2*MARGIN)
            dwg.attribs["preserveAspectRatio"] = "none"    # 歪ませず全面フィット
            # …略…
            
            
    
            if d_parts:
                path = dwg.path(
                    d=" ".join(d_parts),
                    fill=FILL_COLOR,
                    fill_opacity=FILL_OPACITY,
                    stroke=STROKE_COLOR,
                    stroke_width=STROKE_WIDTH,
                    fill_rule="evenodd",
                    id="obj1",
                )
                # ズームしても線幅が太らない（裁ち落としにも強い）
                # path.update({'vector-effect': 'non-scaling-stroke'})
                dwg.add(path)
    
            dwg.saveas(svg_dir / f"mask{i:04}.svg")
    
        self.log.emit(f"[OK] SVGs saved: {svg_dir}")

    

            
    def save_volume_csv(self, nifti_img: nib.Nifti1Image, csv_path: Path, roi_name: str,
                        threshold: float = 0.0, append: bool = False) -> Path:
        """
        ボクセルカウントから体積(mm^3, mL)を算出してCSV出力
        - threshold: 体積カウントの閾値（デフォルト >0）
        - append: Trueなら既存CSVに追記（ヘッダは無ければ書く）
        """
        data = np.asarray(nifti_img.get_fdata())
        # NaNを無視（== False として扱う）
        mask = np.isfinite(data) & (data > threshold)
        voxels = int(np.count_nonzero(mask))
    
        sx, sy, sz = nifti_img.header.get_zooms()[:3]  # mm/voxel
        voxel_mm3 = float(sx * sy * sz)
        vol_mm3 = voxels * voxel_mm3
        vol_ml   = vol_mm3 / 1000.0
    
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        header = [
            "roi", "voxels_count",
            "spacing_x_mm", "spacing_y_mm", "spacing_z_mm",
            "voxel_volume_mm3", "total_volume_mm3", "total_volume_mL"
        ]
        row = [roi_name, voxels, sx, sy, sz, voxel_mm3, vol_mm3, vol_ml]
    
        mode = "a" if append else "w"
        write_header = True
        if append and csv_path.exists():
            # 既存で追記ならヘッダは書かない
            write_header = False
    
        with open(csv_path, mode, newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if write_header:
                w.writerow(header)
            w.writerow(row)
    
        self.log.emit(f"[OK] Volume saved: {csv_path}  ({roi_name}: {vol_ml:.2f} mL)")
        return csv_path
        
    

    def _to_uint8_window(self, arr: np.ndarray) -> np.ndarray:
        """1-99%のロバストウィンドウで 0..255 のuint8へ"""
        vmin, vmax = np.percentile(arr, (1, 99))
        if vmax <= vmin:
            vmin, vmax = float(arr.min()), float(arr.max() or 1.0)
        arr = np.clip(arr, vmin, vmax)
        arr = (arr - vmin) / (vmax - vmin + 1e-8) * 255.0
        return arr.astype(np.uint8)
    
    def save_overlay_jpgs(
        self,
        mask_img: nib.Nifti1Image,
        src_nifti_path: Path,
        out_dir: Path,
        plane: str,
        flip_lr: bool,
        flip_ud: bool,
        reverse_slices: bool,
        alpha: float = 0.7,
    ):
        """
        src_nifti（DICOM→NIfTI化したやつ）にマスクを重ねてJPG出力
        """
        out_dir.mkdir(parents=True, exist_ok=True)
    
        # 向きを揃える
        mask_img = nib.as_closest_canonical(mask_img)
        src_img  = nib.as_closest_canonical(nib.load(str(src_nifti_path)))
    
        m = (np.asarray(mask_img.get_fdata()) > 0.5).astype(np.uint8)
        s = np.asarray(src_img.get_fdata())
    
        ax = {"axial": 2, "coronal": 1, "sagittal": 0}.get(plane, 2)
        n = m.shape[ax]
        order = range(n-1, -1, -1) if reverse_slices else range(n)
    
        # カラー定義
        overlay_rgb = np.array([255, 0, 0], dtype=np.uint8)
    
        for i, k in enumerate(order, start=1):
            m2 = np.take(m, k, axis=ax)
            s2 = np.take(s, k, axis=ax)
    
            if flip_lr:
                m2 = np.flip(m2, axis=1)
                s2 = np.flip(s2, axis=1)
            if flip_ud:
                m2 = np.flip(m2, axis=0)
                s2 = np.flip(s2, axis=0)
    
            base = self._to_uint8_window(s2)                  # (H,W) uint8
            H, W = base.shape
            # 念のためサイズ不一致を吸収
            if m2.shape != (H, W):
                m2 = np.array(
                    Image.fromarray(m2.astype(np.uint8)).resize((W, H), resample=Image.NEAREST),
                    dtype=np.uint8,
                )
    
            base_rgb = np.stack([base, base, base], axis=-1)  # (H,W,3)
            mask_idx = m2.astype(bool)
    
            # αブレンド
            out = base_rgb.copy()
            out[mask_idx] = ( (1.0 - alpha) * base_rgb[mask_idx] + alpha * overlay_rgb ).astype(np.uint8)
    
            Image.fromarray(out).save(out_dir / f"overlay_{i:04}.jpg", quality=95)
    
        self.log.emit(f"[OK] Overlays saved: {out_dir}")
        return out_dir


              
                

            
    
    # def run(self):
    #     try:
    #         self.progress.emit(1)
    
    #         rois = self.selected_rois
    #         # roi_out_name は未使用なので削除してOK（必要なら残しても可）
    #         # roi_out_name = re.sub(r"[^A-Za-z0-9_\-]+", "_", (",".join(rois) if rois else "all").lower())
    
    #         self.log.emit(f"[I] Target ROI(s): {rois if rois else 'ALL'}")
    #         self.job.output_dir.mkdir(parents=True, exist_ok=True)
    
    #         # 1) 入力準備
    #         self.progress.emit(5)
    #         inp_src = Path(self.job.input_path)
    #         out_dir = Path(self.job.output_dir)
    #         tmp_dir = out_dir / "_tmp_ts"
    #         inp_for_ts = ensure_nifti_input(inp_src, tmp_dir)
    
    #         # 2) TS 実行
    #         roi_map = self.run_totalseg(inp_for_ts, out_dir, rois, self.job.use_cpu, self.job.fastest)
    #         self.progress.emit(50)
    
    #         if not roi_map:
    #             raise RuntimeError("No ROI masks were produced by TotalSegmentator.")
    
    #         outputs = []  # 進捗/完了通知用に集計（任意）
    #         # 3) ROIごとに STL / SVG / 体積
    #         for i, (roi, nii_path) in enumerate(roi_map.items(), start=1):
    #             self.log.emit(f"[I] Post-processing ROI: {roi}")
    #             img = nib.load(str(nii_path))
    
    
    
                    
    #             # STL は常に作る
    #             stl_path = out_dir / f"{roi}_mesh.stl"
    #             self.mask_to_stl(img, stl_path, self.job.smooth_iters)
                
    #             # SVG はチェック時のみ
    #             svg_dir = None
    #             if self.job.export_svg:
    #                 svg_dir = out_dir / f"svg_{roi}"
    #                 svg_dir.mkdir(parents=True, exist_ok=True)
    #                 self.mask_to_svgs(
    #                     img, svg_dir,
    #                     plane=self.job.plane,
    #                     reverse_slices=self.job.reverse_slices,
    #                     flip_lr=self.job.flip_lr,
    #                     flip_ud=self.job.flip_ud,
    #                 )
    #             else:
    #                 self.log.emit("[I] SVG export skipped (unchecked).")
                
    #             # CSV はチェック時のみ
    #             vol_csv = None
    #             if self.job.export_csv:
    #                 vol_csv = out_dir / f"volume_{roi}.csv"
    #                 self.save_volume_csv(img, vol_csv, roi)
    #             else:
    #                 self.log.emit("[I] Volume CSV export skipped (unchecked).")
                
    #             outputs.append((roi, nii_path, stl_path, svg_dir, vol_csv))
    
    
    
    
    #             # 進捗
    #             pct = 50 + int(50 * (i / max(1, len(roi_map))))
    #             self.progress.emit(min(pct, 99))
    

    #         self.progress.emit(100)
            
   
    #         if not outputs:
    #             raise RuntimeError("No outputs generated.")
            
    #         roi0, nii0, stl0, svg0, _csv0 = outputs[0]
    #         self.log.emit(f"[I] Returning first ROI to UI: {roi0}")
    #         self.finished.emit(nii0, stl0, svg0)  # svg0 は None の可能性あり                
                
    
    #     except Exception as e:
    #         self.failed.emit(str(e))
                
    def run(self):
        try:
            self.progress.emit(1)
    
    
    
    
            # rois = self.selected_rois
            # self.log.emit(f"[I] Target ROI(s): {rois if rois else 'ALL'}")
            # self.job.output_dir.mkdir(parents=True, exist_ok=True)
            
            # # 1) 入力準備（既存）
            # self.progress.emit(5)
            # inp_src = Path(self.job.input_path)
            # out_dir = Path(self.job.output_dir)
            # tmp_dir = out_dir / "_tmp_ts"
            # inp_for_ts = ensure_nifti_input(inp_src, tmp_dir)
            
            # # ★ 追加：入力のベース名
            # input_base = inp_src.stem if inp_src.is_file() else inp_src.name
                        
            rois = self.selected_rois
            self.log.emit(f"[I] Target ROI(s): {rois if rois else 'ALL'}")
            
            # 入力ソースとベース名
            inp_src = Path(self.job.input_path)
            input_base = inp_src.stem if inp_src.is_file() else inp_src.name
            
            # ★ 出力フォルダ：ユーザー未指定なら <入力名>_Instant3D を使う
            if not self.job.output_dir:
                self.job.output_dir = inp_src.parent / f"{input_base}_Instant3D"
            
            out_dir = Path(self.job.output_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            
            # 1) 入力準備（既存）
            self.progress.emit(5)
            tmp_dir = out_dir / "_tmp_ts"
            inp_for_ts = ensure_nifti_input(inp_src, tmp_dir)
            
            
            
            
            # 2) TS 実行（既存）
            roi_map = self.run_totalseg(inp_for_ts, out_dir, rois, self.job.use_cpu, self.job.fastest)
            self.progress.emit(50)
            
            # 3) ROIごと後処理
            outputs = []
                        
            for i, (roi, nii_path) in enumerate(roi_map.items(), start=1):
                self.log.emit(f"[I] Post-processing ROI: {roi}")
                img = nib.load(str(nii_path))
                canon_img = nib.as_closest_canonical(img)
            
                # --- STL ---
                stl_path = out_dir / f"{input_base}_{roi}_mesh.stl"
                self.mask_to_stl(canon_img, stl_path, self.job.smooth_iters)
            
                # --- SVG ---
                svg_dir = None
                if self.job.export_svg:
                    svg_dir = out_dir / f"svg_{input_base}_{roi}"
                    svg_dir.mkdir(parents=True, exist_ok=True)
                    self.mask_to_svgs(
                        canon_img, svg_dir,
                        plane=self.job.plane,
                        reverse_slices=self.job.reverse_slices,
                        flip_lr=self.job.flip_lr,
                        flip_ud=self.job.flip_ud,
                    )
            
                # --- CSV ---
                vol_csv = None
                if self.job.export_csv:
                    vol_csv = out_dir / f"volume_{input_base}_{roi}.csv"
                    self.save_volume_csv(canon_img, vol_csv, roi)
            
                outputs.append((roi, nii_path, stl_path, svg_dir, vol_csv))
            
            
            # for i, (roi, nii_path) in enumerate(roi_map.items(), start=1):
            #     self.log.emit(f"[I] Post-processing ROI: {roi}")
            #     img = nib.load(str(nii_path))
            
            #     # ★ 個別マスクを input_base 付きで“正規化して保存”
            #     #    （以降 canon_img を下流の STL/SVG/CSV へ統一使用）
            #     canon_img = nib.as_closest_canonical(img)
            #     roi_mask_path = out_dir / f"{input_base}_{roi}.nii.gz"
            #     nib.save(canon_img, str(roi_mask_path))
            
            #     # --- STL（常時） ---
            #     stl_path = out_dir / f"{input_base}_{roi}_mesh.stl"
            #     self.mask_to_stl(canon_img, stl_path, self.job.smooth_iters)
            
            #     # --- SVG（任意） ---
            #     svg_dir = None
            #     if self.job.export_svg:
            #         svg_dir = out_dir / f"svg_{input_base}_{roi}"
            #         svg_dir.mkdir(parents=True, exist_ok=True)
            #         self.mask_to_svgs(
            #             canon_img, svg_dir,
            #             plane=self.job.plane,
            #             reverse_slices=self.job.reverse_slices,
            #             flip_lr=self.job.flip_lr,
            #             flip_ud=self.job.flip_ud,
            #         )
            #     else:
            #         self.log.emit("[I] SVG export skipped (unchecked).")
            
            #     # --- CSV（任意） ---
            #     vol_csv = None
            #     if self.job.export_csv:
            #         vol_csv = out_dir / f"volume_{input_base}_{roi}.csv"
            #         self.save_volume_csv(canon_img, vol_csv, roi)
            #     else:
            #         self.log.emit("[I] Volume CSV export skipped (unchecked).")
            
            #     # ★ UI返却用にも “新しいNIfTIパス” を使う
            #     outputs.append((roi, roi_mask_path, stl_path, svg_dir, vol_csv))
                
                
                
                
                
            
                pct = 50 + int(50 * (i / max(1, len(roi_map))))
                self.progress.emit(min(pct, 99))
            
            self.progress.emit(100)
            if not outputs:
                raise RuntimeError("No outputs generated.")
            roi0, nii0, stl0, svg0, _csv0 = outputs[0]
            self.log.emit(f"[I] Returning first ROI to UI: {roi0}")
            self.finished.emit(nii0, stl0, svg0)
    
            # # 1) 入力準備
            # self.progress.emit(5)
            # inp_src = Path(self.job.input_path)
            # out_dir = Path(self.job.output_dir)
            # tmp_dir = out_dir / "_tmp_ts"
            # inp_for_ts = ensure_nifti_input(inp_src, tmp_dir)
    
            # # 入力ファイル／フォルダ名のベースを取得 ← ★追加
            # input_base = inp_src.stem if inp_src.is_file() else inp_src.name
    
            # # 2) TS 実行
            # roi_map = self.run_totalseg(inp_for_ts, out_dir, rois, self.job.use_cpu, self.job.fastest)
            # self.progress.emit(50)
    
            # if not roi_map:
            #     raise RuntimeError("No ROI masks were produced by TotalSegmentator.")
    
            # outputs = []  
            # # 3) ROIごとに STL / SVG / 体積
            # for i, (roi, nii_path) in enumerate(roi_map.items(), start=1):
            #     self.log.emit(f"[I] Post-processing ROI: {roi}")
            #     img = nib.load(str(nii_path))
    
            #     # --- STL ---
            #     stl_path = out_dir / f"{input_base}_{roi}_mesh.stl"
            #     self.mask_to_stl(img, stl_path, self.job.smooth_iters)
    
            #     # --- SVG ---
            #     svg_dir = None
            #     if self.job.export_svg:
            #         svg_dir = out_dir / f"svg_{input_base}_{roi}"
            #         svg_dir.mkdir(parents=True, exist_ok=True)
            #         self.mask_to_svgs(
            #             img, svg_dir,
            #             plane=self.job.plane,
            #             reverse_slices=self.job.reverse_slices,
            #             flip_lr=self.job.flip_lr,
            #             flip_ud=self.job.flip_ud,
            #         )
            #     else:
            #         self.log.emit("[I] SVG export skipped (unchecked).")
    
            #     # --- CSV ---
            #     vol_csv = None
            #     if self.job.export_csv:
            #         vol_csv = out_dir / f"volume_{input_base}_{roi}.csv"
            #         self.save_volume_csv(img, vol_csv, roi)
            #     else:
            #         self.log.emit("[I] Volume CSV export skipped (unchecked).")
    
            #     outputs.append((roi, nii_path, stl_path, svg_dir, vol_csv))
    
            #     # 進捗
            #     pct = 50 + int(50 * (i / max(1, len(roi_map))))
            #     self.progress.emit(min(pct, 99))
    
            # self.progress.emit(100)
    
            # if not outputs:
            #     raise RuntimeError("No outputs generated.")
    
            # roi0, nii0, stl0, svg0, _csv0 = outputs[0]
            # self.log.emit(f"[I] Returning first ROI to UI: {roi0}")
            # self.finished.emit(nii0, stl0, svg0)
    
        except Exception as e:
            self.failed.emit(str(e))
            


# ----------------- GUI -----------------

class MainWindow(QWidget):

    
    
    
    def __init__(self):
        super().__init__()
        
        # self.ts_home = Path(sys.argv[0]).parent / "ts_home"
        # os.environ["TOTALSEG_HOME_DIR"] = str(self.ts_home)
        # self.ts_home.mkdir(parents=True, exist_ok=True)
        
        

        self.setWindowTitle("Instant3D — Fast STL from DICOM")
        self.resize(780, 620)
        
        self.chk_batch = QCheckBox("Batch process (treat input as a folder of cases)")
        self.chk_batch.setChecked(False)

        self.input_edit = QLineEdit()
        self.input_btn = QPushButton("Browse…")
        self.input_btn.clicked.connect(self.on_browse_input)
        # self.input_edit.setProperty("accentField", True)
        self.input_btn.setProperty("accentButton", True)    
        

        
        
        
        self.out_edit = QLineEdit()
        self.out_btn = QPushButton("Browse…")
        self.out_btn.clicked.connect(self.on_browse_out)




        self.modality_combo = QComboBox()
        self.modality_combo.addItems(["CT", "MRI"])
        self.modality_combo.setCurrentText("CT")
        
                
                
        self.task_combo = QComboBox()
        self.task_combo.addItem("Auto (by ROI/modality)", userData="__auto__")
        self.task_combo.insertSeparator(self.task_combo.count())
        self.task_combo.addItem("— Open tasks —", userData=None)
        for t in OPEN_TASKS:
            self.task_combo.addItem(t, userData=t)
        self.task_combo.insertSeparator(self.task_combo.count())
        self.task_combo.addItem("— Licensed tasks —", userData=None)
        for t in LICENSED_TASKS:
            self.task_combo.addItem("🔒 " + t, userData=t)
        
        # ▼ここを置き換え（Auto をデフォルト選択）
        i = self.task_combo.findData("__auto__")
        if i != -1:
            self.task_combo.setCurrentIndex(i)
        else:
            self.task_combo.setCurrentIndex(0)
                

        
        
        
        
        
        self.chk_robust = QCheckBox("Robust crop (3mm, safer for ROI subset)")
        self.chk_robust.setChecked(False)
        
        
        
        
        # Hook up ROI suggestions to the selected task
        self.task_combo.currentIndexChanged.connect(self._on_task_changed)        
        
        
        

        # ▼ ROI 入力 + サジェスト（前回どおり）
        self.roi_input = QLineEdit()
        self.roi_input.setPlaceholderText("Type ROI name (e.g., liver, kidney_left). Press 'Add ROI' to add.")
        self.roi_input.setProperty("accentField", True)
        
        self._roi_all = load_roi_catalog(self.modality_combo.currentText())
        self._roi_model = QStringListModel(self._roi_all)
        self._roi_completer = QCompleter(self._roi_model, self)
        self._roi_completer.setCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
        self._roi_completer.setFilterMode(Qt.MatchFlag.MatchContains)
        self.roi_input.setCompleter(self._roi_completer)
        self.roi_input.textChanged.connect(self._on_roi_text_changed)
        # Enterキーで ROI を追加するショートカット
        self.roi_input.returnPressed.connect(self._on_add_roi)

        

        self.btn_add_roi = QPushButton("Add ROI")
        self.btn_add_roi.clicked.connect(self._on_add_roi)
        self.btn_add_roi.setProperty("accentButton", True)
        self.roi_list = QListWidget()
        self.roi_list.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        self.roi_list.itemDoubleClicked.connect(lambda it: self._remove_selected())
        self.roi_list.setMaximumHeight(100)
  
        
        # 追加/削除ボタン
        self.btn_remove_roi = QPushButton("Remove selected")
        self.btn_remove_roi.clicked.connect(self._remove_selected)

        self.btn_clear_rois = QPushButton("Clear all")
        self.btn_clear_rois.clicked.connect(self._clear_all_rois)

        # Deleteキーで選択削除
        self.del_shortcut = QShortcut(QKeySequence(Qt.Key.Key_Delete), self.roi_list)
        self.del_shortcut.activated.connect(self._remove_selected)
        

        # ▼ 下部：全ROI表示 + 簡易検索 + コピー
        self.allroi_search = QLineEdit()
        self.allroi_search.setPlaceholderText("Filter the full list below...")
        self.allroi_search.textChanged.connect(self._on_allroi_filter)

        self.allroi_view = QTextEdit()
        self.allroi_view.setReadOnly(True)
        self.btn_copy_all = QPushButton("Copy full list")
        self.btn_copy_all.clicked.connect(self._copy_allroi_to_clipboard)
        
        self._roi_task_index = {}
        self._build_roi_task_index()
        self._render_allroi_view()  # 初回描画を「ROI — tasks」に


        # # 初期描画
        # self._render_allroi_view(self._roi_all)





        # plane + flips
        self.plane_combo = QComboBox(); self.plane_combo.addItems(["axial", "coronal", "sagittal"]) ; self.plane_combo.setCurrentText("axial")
        self.chk_reverse = QCheckBox("Reverse slice order")
        self.chk_flip_lr = QCheckBox("Flip L/R")
        self.chk_flip_ud = QCheckBox("Flip U/D")
                    
        

        # perf
        self.chk_cpu = QCheckBox("Force CPU")
        # self.chk_fastest = QCheckBox("Fastest (6mm resample)"); self.chk_fastest.setChecked(True)
        # self.chk_fastest = QCheckBox("Fastest (6mm resample)")
        self.chk_fastest = QCheckBox("Fast (3mm model)")
        self.chk_fastest.setChecked(False)   # ← ここだけOFFに
        self.smooth_spin = QSpinBox(); self.smooth_spin.setRange(0, 100); self.smooth_spin.setValue(10)
                
        self.chk_export_svg = QCheckBox("Export SVG masks")
        self.chk_export_svg.setChecked(False)  # デフォルトOFF
        
        self.chk_export_csv = QCheckBox("Export volume CSV")
        self.chk_export_csv.setChecked(False)  # デフォルトOFF        

        self.run_btn = QPushButton("Run")
        self.run_btn.clicked.connect(self.on_run)
        self.run_btn.setProperty("primary", True)
        self.run_btn.setShortcut("Ctrl+Return")   # 任意: ショートカット
        self.run_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.input_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_add_roi.setCursor(Qt.CursorShape.PointingHandCursor)        
        
        self.open_btn = QPushButton("Open output folder")
        self.open_btn.clicked.connect(self.on_open_out)

        self.log = QTextEdit(); self.log.setReadOnly(True)
        self.progress = QProgressBar(); self.progress.setRange(0, 100)


        # 文字を太字にしたいボタンだけ
        for btn in (self.input_btn, self.btn_add_roi, self.run_btn):
            f = btn.font()
            f.setBold(True)
            btn.setFont(f)
                
        # ---- ここは Layouts より前で作っておく（強調したいので QLabel に） ----
        lbl_input    = QLabel("Input (DICOM folder or NIfTI):");      lbl_input.setProperty("accentLabel", True)
        # lbl_output   = QLabel("Output folder (default: <input>/Instant3DSAVE):")
        lbl_output   = QLabel("Output folder (default: <input>_Instant3D):")
        lbl_modality = QLabel("Imaging modality:");                    lbl_modality.setProperty("accentLabel", True)
        lbl_task     = QLabel("Segmentation task:");                   lbl_task.setProperty("accentLabel", True)
        lbl_roi      = QLabel("ROI:");                                 lbl_roi.setProperty("accentLabel", True)
        
        # ----------------- GUI -----------------
        # Layouts
        form = QFormLayout()
        
        form.addRow("Batch:", self.chk_batch)
        
        # 入力行（ラベルだけの行 → 次行でh1を丸ごと1行として追加）
        h1 = QHBoxLayout(); h1.addWidget(self.input_edit); h1.addWidget(self.input_btn)
        h2 = QHBoxLayout(); h2.addWidget(self.out_edit);   h2.addWidget(self.out_btn)
        
        form.addRow(lbl_input, QWidget())    # ← ラベルのみ（右は空のプレースホルダ）
        form.addRow(h1)                      # ← 実際の入力行（1行まるごと）
        
        form.addRow(lbl_output, QWidget())   # 出力フォルダのラベル
        form.addRow(h2)                      # 出力フォルダの実入力行
        
        # モダリティ／タスク（文字列ラベルの addRow は削除し、QLabel 版だけ残す）
        form.addRow(lbl_modality, self.modality_combo)
        form.addRow(lbl_task, self.task_combo)
        
        # lic_box = QHBoxLayout()
        # lic_box.addWidget(self.btn_set_license)
        # lic_box.addWidget(self.lbl_license_status)
        # lic_box.addStretch(1)
        # lic_w = QWidget(); lic_w.setLayout(lic_box)
        # form.addRow("", lic_w)        
        
        
        form.addRow("", self.chk_robust)     # タスクの下にチェックをぶら下げ
        
        # ROI（こちらも重複をなくして QLabel 版だけ）
        form.addRow(lbl_roi, self.roi_input)
        
        
        
        
        
        
        roi_btns = QHBoxLayout()
        roi_btns.addWidget(self.btn_add_roi)
        roi_btns.addWidget(self.btn_remove_roi)
        roi_btns.addWidget(self.btn_clear_rois)
        roi_btns.addStretch()
        roi_btns_w = QWidget(); roi_btns_w.setLayout(roi_btns)
        form.addRow("", roi_btns_w)
        
        form.addRow("Selected ROIs:", self.roi_list)
        
        
        form.addRow("Plane:", self.plane_combo)
        flips_box = QHBoxLayout(); flips_box.addWidget(self.chk_reverse); flips_box.addWidget(self.chk_flip_lr); flips_box.addWidget(self.chk_flip_ud)
        flips_w = QWidget(); flips_w.setLayout(flips_box)
        form.addRow("Slice order & flips:", flips_w)
        
        # ★ NRRDヒントラベルを追加
        lbl_nrrd_tip = QLabel("Tip: For NRRD files, please check both Flip L/R and Flip U/D.")
        lbl_nrrd_tip.setWordWrap(True)
        font = lbl_nrrd_tip.font()
        font.setItalic(True)
        font.setPointSize(max(font.pointSize() - 1, 9))
        lbl_nrrd_tip.setFont(font)
        lbl_nrrd_tip.setStyleSheet("color: #6b7280; margin-left: 2px;")
        form.addRow("", lbl_nrrd_tip)  # ラベルの説明行だけ出す        
        
        perf_box = QHBoxLayout(); perf_box.addWidget(self.chk_cpu); perf_box.addWidget(self.chk_fastest)
        perf_w = QWidget(); perf_w.setLayout(perf_box)
        form.addRow("Performance:", perf_w)
        form.addRow("Mesh smoothing (iters):", self.smooth_spin)
             
        
        # form.addRow("Outputs:", out_row)
        out_box = QHBoxLayout()
        out_box.addWidget(self.chk_export_svg)
        out_box.addWidget(self.chk_export_csv)
        # 同じ見た目にしたいので特に stretch / spacing は入れない
        out_w = QWidget(); out_w.setLayout(out_box)
        form.addRow("Outputs:", out_w)


        btns = QHBoxLayout(); btns.addWidget(self.run_btn); btns.addWidget(self.open_btn); btns.addStretch()

        root = QVBoxLayout(self)
        
        grp = QGroupBox("Settings"); grp.setLayout(form)
        root.addWidget(grp)
        root.addLayout(btns)
        root.addWidget(QLabel("Log:"))
        root.addWidget(self.log, 1)
        root.addWidget(self.progress)
        
        # ▼ 全リスト（下部）を既存 root に追加（※ root を新しく作り直さない）
        vbox_bottom = QVBoxLayout()
        vbox_bottom.addWidget(self.allroi_search)
        vbox_bottom.addWidget(self.allroi_view)
        vbox_bottom.addWidget(self.btn_copy_all)
        root.addLayout(vbox_bottom)      

        self.worker: Worker | None = None
        
        self.modality_combo.currentTextChanged.connect(self.on_modality_changed)
        
        self._on_task_changed()
        






            
    def _build_roi_task_index(self):
        """ROI -> [tasks] の逆引きインデックスを構築"""
        idx: dict[str, list[str]] = {}
        # どのタスクを対象にするか（定義済みのリストを利用）
        all_tasks = list(OPEN_TASKS) + list(LICENSED_TASKS)
    
        for t in all_tasks:
            rois = load_roi_catalog_for_task(t, self.modality_combo.currentText())
            if not rois:
                continue
            for r in rois:
                r = r.strip()
                if not r:
                    continue
                idx.setdefault(r, []).append(t)
    
        # タスク名は見やすいようにソート
        for r in idx:
            idx[r].sort()
    
        self._roi_task_index = idx
    
    
    def _on_task_changed(self):
        userTask = self.task_combo.currentData()
        # ▼ ROI候補のソースを切り替え
        if userTask and userTask != "__auto__":
            rois = load_roi_catalog_for_task(userTask, self.modality_combo.currentText())
        else:
            rois = load_roi_catalog(self.modality_combo.currentText())
        rois = list(rois)                 # ← 念のため明示的に list 化    
        self._roi_all = rois
        self._roi_model.setStringList(self._roi_all)
    
        # ▼ 選択済みROIの整合を取り直し
        valid = set(self._roi_all)
        to_remove = [i for i in range(self.roi_list.count())
                     if self.roi_list.item(i).text() not in valid and userTask not in (None, "__auto__")]
        for offset, idx in enumerate(to_remove):
            self.roi_list.takeItem(idx - offset)
    
        # ▼ Auto以外ならモダリティUIを無効化（説明用にツールチップも）
        self.modality_combo.setEnabled(userTask in (None, "__auto__"))
        self.modality_combo.setToolTip(
            "Used only when Segmentation task = Auto (CT→total / MRI→total_mr)."
        )
        self._build_roi_task_index()
        self._render_allroi_view()



    def _ensure_valid_task_selection(self):
        i = self.task_combo.currentIndex()
        data = self.task_combo.itemData(i)
        if data is None:  # 見出しや無効行に当たったら
            # 次の有効項目へ
            for j in range(i + 1, self.task_combo.count()):
                if self.task_combo.itemData(j) is not None:
                    self.task_combo.setCurrentIndex(j)
                    return
            # 予備：前方へ
            for j in range(i - 1, -1, -1):
                if self.task_combo.itemData(j) is not None:
                    self.task_combo.setCurrentIndex(j)
                    return







    # ---------- helpers ----------
    
    def _clear_all_rois(self):
        self.roi_list.clear()
    
        
    def _render_allroi_view(self, rois: list[str] | None = None):
        """下部ビューを 'ROI — tasks' 形式で描画"""
        # フォントは等幅にすると揃って見える（任意）
        try:
            self.allroi_view.setFontFamily("monospace")
        except Exception:
            pass
    
        # ライセンス有無（未定義なら True 扱い＝鍵を出さない）
        try:
            licensed_ok = has_totalseg_license()
        except Exception:
            licensed_ok = True
    
        if rois is None:
            rois = sorted(self._roi_task_index.keys())
        else:
            rois = sorted(rois)
    
        lines = []
        lock = "🔒"
        licensed_set = set(LICENSED_TASKS)
    
        # 幅を軽く整える（左カラム幅を計算）
        left_width = 1
        for r in rois:
            if len(r) > left_width:
                left_width = len(r)
        left_width = min(max(left_width, 8), 36)
    
        for r in rois:
            tasks = self._roi_task_index.get(r, [])
            if not tasks:
                continue
            tags = []
            for t in tasks:
                if (t in licensed_set) and (not licensed_ok):
                    tags.append(f"{t}{lock}")
                else:
                    tags.append(t)
            left = f"{r:<{left_width}}"
            lines.append(f"{left} — {', '.join(tags)}")
    
        self.allroi_view.setPlainText("\n".join(lines))

            

        
    def _on_allroi_filter(self, text: str):
        q = (text or "").strip().lower()
        if not q:
            self._render_allroi_view()
            return
    
        if q.startswith("task:"):
            tq = q[5:].strip()
            # タスク名に tq を含むものに属する ROI を全部表示
            rois = [r for r, tasks in self._roi_task_index.items()
                    if any(tq in t.lower() for t in tasks)]
        else:
            # 既定: ROI 名でフィルタ
            rois = [r for r in self._roi_task_index.keys() if q in r.lower()]
    
        self._render_allroi_view(rois)


        
        

    def _copy_allroi_to_clipboard(self):
        cb = QApplication.clipboard()
        cb.setText(self.allroi_view.toPlainText())
        self.append_log("[I] Full ROI list copied to clipboard.")


    def _on_roi_text_changed(self, text: str):
        topk = fuzzy_topk(text, list(self._roi_all), k=30)  # ← list(...) を追加
        # topk = fuzzy_topk(text, self._roi_all, k=30)
        self._roi_model.setStringList(topk)





    # def _on_add_roi(self):
    #     text = self.roi_input.text().strip()
    #     if not text:
    #         return
    #     targets = ROI_ALIAS.get(text, [text])
    #     valid = set(self._roi_all)
    #     added, missing = [], []
    #     for t in targets:
    #         if t in valid:
    #             if not self._exists_in_list(self.roi_list, t):
    #                 self.roi_list.addItem(QListWidgetItem(t))
    #                 added.append(t)
    #         else:
    #             missing.append(t)
    #     if missing:
    #         QMessageBox.warning(self, "Unsupported ROI",
    #                             f"{', '.join(missing)} is not available for {self.modality_combo.currentText()} task.")
    #     if added:
    #         self.roi_input.clear()
    

    
    def _on_add_roi(self):
        text = self.roi_input.text().strip()
        if not text:
            return
        name = ROI_SYNONYM.get(text, text)          # 同義語を正規化
        targets = ROI_GROUPS.get(name, [name])      # グループ展開
        valid = set(self._roi_all)
    
        added, missing = [], []
        for t in targets:
            if t in valid and not self._exists_in_list(self.roi_list, t):
                self.roi_list.addItem(QListWidgetItem(t))
                added.append(t)
            else:
                if t not in valid:
                    missing.append(t)
    
        if missing:
            QMessageBox.warning(self, "Unsupported ROI",
                f"{', '.join(missing)} is not available for the current task.")
        if added:
            self.roi_input.clear()




    def _exists_in_list(self, listw: QListWidget, text: str) -> bool:
        return any(listw.item(i).text() == text for i in range(listw.count()))

    def _remove_selected(self):
        for it in self.roi_list.selectedItems():
            self.roi_list.takeItem(self.roi_list.row(it))





    # def on_modality_changed(self, modality: str):
    #     # 候補と全リストを差し替え
    #     self._roi_all = load_roi_catalog(modality)
    #     self._roi_model.setStringList(self._roi_all)
    #     self.roi_input.clear()
    #     # 既選択の互換チェック（未対応は落とす）
    #     keep = []
    #     for i in range(self.roi_list.count()):
    #         t = self.roi_list.item(i).text()
    #         if t in self._roi_all:
    #             keep.append(t)
    #     self.roi_list.clear()
    #     for t in keep:
    #         self.roi_list.addItem(QListWidgetItem(t))
    #     # 下部ビュー再描画
    #     self._render_allroi_view(self._roi_all)
    #     self.allroi_search.clear()

    
    def on_modality_changed(self, modality: str):
        # 明示タスク時は無視（ROI候補はタスクで決まる）
        if self.task_combo.currentData() not in (None, "__auto__"):
            return
    
        # Autoのときだけ従来どおり更新
        self._roi_all = load_roi_catalog(modality)
        self._roi_model.setStringList(self._roi_all)
        self.roi_input.clear()
        keep = [self.roi_list.item(i).text() for i in range(self.roi_list.count())
                if self.roi_list.item(i).text() in self._roi_all]
        self.roi_list.clear()
        for t in keep:
            self.roi_list.addItem(QListWidgetItem(t))
        self._render_allroi_view(self._roi_all)
        self.allroi_search.clear()
        
        self._build_roi_task_index()
        self._render_allroi_view()






    
    # def ensure_nifti_input(src_path: Path, tmp_dir: Path) -> Path:
    #     """
    #     Return a path to a NIfTI (.nii.gz) file that TotalSegmentator can read.
    #     If src is DICOM folder -> return folder as-is.
    #     If src is .nii/.nii.gz -> return file as-is.
    #     If src is .nrrd -> convert to .nii.gz in tmp_dir and return that path.
    #     """
    #     if src_path.is_dir():
    #         return src_path  # DICOM series
    
    #     ext = src_path.suffix.lower()
    #     if ext in [".nii", ".gz"] or src_path.name.endswith(".nii.gz"):
    #         return src_path  # already nifti
    
    #     if ext == ".nrrd":
    #         tmp_dir.mkdir(parents=True, exist_ok=True)
    #         out_nii = tmp_dir / "input_from_nrrd.nii.gz"
    #         # Prefer SimpleITK (keeps spacing/origin/direction safely)
    #         try:
    #             import SimpleITK as sitk
    #             img = sitk.ReadImage(str(src_path))
    #             sitk.WriteImage(img, str(out_nii))
    #             return out_nii
    #         except Exception:
    #             # Fallback: pynrrd + nibabel (best-effort)
    #             import nrrd, nibabel as nib, numpy as np
    #             data, hdr = nrrd.read(str(src_path))
    #             # Build affine from space directions / origin if available
    #             dirs = hdr.get("space directions")
    #             origin = hdr.get("space origin", [0, 0, 0])
    #             aff = np.eye(4)
    #             try:
    #                 # dirs may be a 3x3 of direction*spacing vectors
    #                 M = np.array([[d[0], d[1], d[2]] for d in dirs], dtype=float)
    #                 aff[:3, :3] = M
    #             except Exception:
    #                 pass
    #             try:
    #                 aff[:3, 3] = np.array(origin, dtype=float)
    #             except Exception:
    #                 pass
    #             img = nib.Nifti1Image(np.asarray(data), aff)
    #             nib.save(img, str(out_nii))
    #             return out_nii
    
    #     raise ValueError(f"Unsupported input type: {src_path}")






    # -------- UI handlers --------

    
    
    # def on_browse_input(self):
    #     base = str(Path.home())
    
    #     # ① まずファイル（NIfTI/NRRD/単枚DICOM）を見せる
    #     file, _ = QFileDialog.getOpenFileName(
    #         self,
    #         "Select volume file (NIfTI / NRRD / single DICOM)",
    #         base,
    #         "All supported (*.nii *.nii.gz *.nrrd *.nrrd.gz *.nhdr *.dcm);;"
    #         "NIfTI (*.nii *.nii.gz);;"
    #         "NRRD (*.nrrd *.nrrd.gz *.nhdr);;"
    #         "DICOM (*.dcm);;"
    #         "All files (*.*)"
    #     )
    
    #     path = file
    
    #     # ② キャンセルならフォルダ（DICOMシリーズ）を選ばせる
    #     if not path:
    #         folder = QFileDialog.getExistingDirectory(self, "Select DICOM folder", base)
    #         path = folder
    
    #     if not path:
    #         return
    
    #     # ③ テキスト欄に反映／出力先を自動設定
    #     self.input_edit.setText(path)
    #     in_path = Path(path)
                
    #     input_base = in_path.stem if in_path.is_file() else in_path.name
    #     out_dir = (in_path.parent / f"{input_base}_Instant3D") if in_path.is_file() \
    #               else (in_path.parent / f"{input_base}_Instant3D")
    #     self.out_edit.setText(str(out_dir))
        
        
    #     # out_dir = (in_path / "Instant3DSAVE") if in_path.is_dir() \
    #     #           else (in_path.parent / "Instant3DSAVE")
    #     # self.out_edit.setText(str(out_dir))
            
    # def on_browse_input(self):
    #     base = str(Path.home())
    
    #     path = ""
    #     if self.chk_batch.isChecked():
    #         folder = QFileDialog.getExistingDirectory(self, "Select parent folder (contains cases)", base)
    #         path = folder or ""
    #     else:
    #         file, _ = QFileDialog.getOpenFileName(
    #             self, "Select volume file (NIfTI / NRRD / single DICOM)",
    #             base,
    #             "All supported (*.nii *.nii.gz *.nrrd *.nrrd.gz *.nhdr *.dcm);;"
    #             "NIfTI (*.nii *.nii.gz);;NRRD (*.nrrd *.nrrd.gz *.nhdr);;DICOM (*.dcm);;All files (*.*)"
    #         )
    #         path = file
    #         if not path:
    #             folder = QFileDialog.getExistingDirectory(self, "Select DICOM folder", base)
    #             path = folder or ""
    
    #     if not path:
    #         return
    
    #     self.input_edit.setText(path)
    #     in_path = Path(path)
    #     input_base = in_path.stem if in_path.is_file() else in_path.name
    #     # デフォルト出力：<入力名>_Instant3D
    #     out_dir = in_path.parent / f"{input_base}_Instant3D"
    #     self.out_edit.setText(str(out_dir))
            
    def on_browse_input(self):
        base = str(Path.home())
    
        path = ""
        if self.chk_batch.isChecked():
            folder = QFileDialog.getExistingDirectory(self, "Select parent folder (contains cases)", base)
            path = folder or ""
        else:
            file, _ = QFileDialog.getOpenFileName(
                self, "Select volume file (NIfTI / NRRD / single DICOM)",
                base,
                "All supported (*.nii *.nii.gz *.nrrd *.nrrd.gz *.nhdr *.dcm);;"
                "NIfTI (*.nii *.nii.gz);;NRRD (*.nrrd *.nrrd.gz *.nhdr);;DICOM (*.dcm);;All files (*.*)"
            )
            path = file
            if not path:
                folder = QFileDialog.getExistingDirectory(self, "Select DICOM folder", base)
                path = folder or ""
    
        if not path:
            return
    
        self.input_edit.setText(path)
        in_path = Path(path)
    
        # ▼ バッチONなら説明テキストを出して終了
        if self.chk_batch.isChecked():
            self.out_edit.setText("<auto: each case → <case>_Instant3D>")
            return
    
        # ▼ 単体モード：<入力名>_Instant3D を提案
        input_base = in_path.stem if in_path.is_file() else in_path.name
        out_dir = in_path.parent / f"{input_base}_Instant3D"
        self.out_edit.setText(str(out_dir))
        


    def on_browse_out(self):
        base = str(Path.home())
        folder = QFileDialog.getExistingDirectory(self, "Select output folder", base)
        if folder:
            self.out_edit.setText(folder)

    def on_open_out(self):
        out_dir = self.out_edit.text().strip()
        if not out_dir:
            QMessageBox.information(self, "Info", "Output folder is empty.")
            return
        d = Path(out_dir)
        if not d.exists():
            QMessageBox.information(self, "Info", "Output folder does not exist yet.")
            return
        # Open in OS file browser
        if sys.platform.startswith("win"):
            os.startfile(d)  # type: ignore
        elif sys.platform == "darwin":
            subprocess.run(["open", str(d)])
        else:
            subprocess.run(["xdg-open", str(d)])

    # def append_log(self, text: str):
    #     self.log.append(text)
    #     self.log.moveCursor(self.log.textCursor().End)
        
    def append_log(self, msg: str):
        self.log.append(msg)
        # ↓ PyQt6 正式表記
        self.log.moveCursor(QTextCursor.MoveOperation.End)
        self.log.ensureCursorVisible()        
        
        



    
    # def on_run(self):
    #     # tdata = self.task_combo.currentData()
    #     # if tdata in LICENSED_TASKS and not has_totalseg_license():
    #     #     QMessageBox.warning(
    #     #         self, "License required",
    #     #         f"Task '{tdata}' requires a license.\n"
    #     #         "Press 'Set License…' to register your key."
    #     #     )
    #     #     return     
        
    #     # on_run() 冒頭で
    #     try:
    #         exe = which_totalseg()
    #     except FileNotFoundError as e:
    #         QMessageBox.warning(self, "TotalSegmentator not found", str(e))
    #         return
        
    #     # ライセンス必須タスクの時だけ確認（研究/商用どちらでもキー必要）
    #     tdata = self.task_combo.currentData()
    #     if tdata in LICENSED_TASKS:
    #         ok_path = Path.home() / ".totalsegmentator" / "config.json"
    #         if not ok_path.exists():
    #             QMessageBox.warning(
    #                 self, "License required",
    #                 "This task requires a TotalSegmentator license.\n\n"
    #                 "Open Command Prompt and run:\n"
    #                 "  totalseg_set_license -l <YOUR_KEY>\n\n"
    #                 "Then click Run again."
    #             )
    #             return
        
        
    #     try:
    #         inp = self.input_edit.text().strip()
    #         if not inp:
    #             QMessageBox.warning(self, "Missing input", "Please select a DICOM folder or a NIfTI file.")
    #             return
    #         in_path = Path(inp)
    #         if not in_path.exists():
    #             QMessageBox.warning(self, "Invalid input", "Selected input path does not exist.")
    #             return
    
    #         # out_dir = self.out_edit.text().strip()
    #         # out_dir = out_dir or str((in_path if in_path.is_dir() else in_path.parent) / "Instant3DSAVE")
    #         out_dir = self.out_edit.text().strip()
    #         if not out_dir:
    #             input_base = in_path.stem if in_path.is_file() else in_path.name
    #             # 入力がファイルでもフォルダでも、親フォルダの直下に <入力名>_Instant3D を作る
    #             out_dir = str(in_path.parent / f"{input_base}_Instant3D")
            
            
    
    
    
    
    
    #         # # --- ROI収集（複数） ------------------------------------------ # NEW
    #         # # すでに選択リストにあるものを取得。未追加で入力欄にあるだけなら一度追加扱いにする
    #         # rois = [self.roi_list.item(i).text() for i in range(self.roi_list.count())]
    #         # if not rois and self.roi_input.text().strip():
    #         #     self._on_add_roi()  # 入力欄の内容をリストに反映
    #         #     rois = [self.roi_list.item(i).text() for i in range(self.roi_list.count())]
    
    #         # if not rois:
    #         #     QMessageBox.warning(self, "No ROI", "Please add at least one ROI.")
    #         #     return
    #         # # ------------------------------------------------------------- # NEW
                        
    #         # # …（前段のライセンスチェック・入力/ROIチェックはそのまま）…
            
    #         # ROI収集（既存）
    #         rois = [self.roi_list.item(i).text() for i in range(self.roi_list.count())]
    #         if not rois and self.roi_input.text().strip():
    #             self._on_add_roi()
    #             rois = [self.roi_list.item(i).text() for i in range(self.roi_list.count())]
    #         if not rois:
    #             QMessageBox.warning(self, "No ROI", "Please add at least one ROI.")
    #             return
            
    #         # ★ ここから追加：バッチ分岐
    #         if self.chk_batch.isChecked():
    #             if not in_path.is_dir():
    #                 QMessageBox.warning(self, "Batch input required", "In batch mode, please select a parent folder.")
    #                 return
            
    #             SUP = {".nii", ".nii.gz", ".nrrd", ".nhdr", ".dcm"}
    #             children = sorted(in_path.iterdir(), key=lambda p: p.name.lower())
    #             targets = []
    #             for p in children:
    #                 if p.is_dir():
    #                     targets.append(p)
    #                 elif p.is_file():
    #                     ext = (".nii.gz" if p.name.lower().endswith(".nii.gz") else p.suffix.lower())
    #                     if ext in SUP:
    #                         targets.append(p)
            
    #             if not targets:
    #                 QMessageBox.information(self, "No cases found",
    #                                         "No subfolders or supported files were found directly under the selected folder.")
    #                 return
            
    #             # バッチ用状態をセット
    #             self._batch_targets = targets
    #             self._batch_index = 0
    #             self._batch_rois = rois
            
    #             self.log.clear()
    #             self.progress.setValue(0)
    #             self.append_log(f"[I] Batch: {len(targets)} case(s) found under: {in_path}")
    #             self.setEnabled(False)
    #             self._run_next_in_batch()   # ★ ここで1件目起動
    #             return
    #         # ★ バッチでなければ、従来の単体実行へ続く
            
            
            
            
                        
    #         out_dir = self.out_edit.text().strip()
    #         if not out_dir:
    #             input_base = in_path.stem if in_path.is_file() else in_path.name
    #             out_dir = str(in_path.parent / f"{input_base}_Instant3D")
    #         Path(out_dir).mkdir(parents=True, exist_ok=True)
            
            
            
            
    
    
    #         job = Job(
    #             input_path=in_path,
    #             output_dir=Path(out_dir),
    #             # 互換のために何か文字列を渡しておく（使わないなら Job から roi_name を外してOK）
    #             roi_name=",".join(rois),                                            # NEW (旧: self.roi_combo.currentText())
    #             plane=self.plane_combo.currentText(),
    #             reverse_slices=self.chk_reverse.isChecked(),
    #             flip_lr=self.chk_flip_lr.isChecked(),
    #             flip_ud=self.chk_flip_ud.isChecked(),
    #             use_cpu=self.chk_cpu.isChecked(),
    #             fastest=self.chk_fastest.isChecked(),
    #             smooth_iters=int(self.smooth_spin.value()),
    #             modality=self.modality_combo.currentText(),   # 既出
    #             task=(self.task_combo.currentData() or "__auto__"),
    #             robust_crop=self.chk_robust.isChecked(),
    #             export_svg=self.chk_export_svg.isChecked(),
    #             export_csv=self.chk_export_csv.isChecked(),                
    #         )
    
    #         # kick worker
    #         self.log.clear()
    #         self.progress.setValue(0)
    #         self.append_log("[I] Starting…")
    
    #         # 複数ROIを Worker に渡す                                                # NEW
    #         self.worker = Worker(job, selected_rois=rois)                            # NEW
    #         # （Worker.run_totalseg 内で args = [..., "--roi_subset"] + selected_rois になっている想定）
    
    #         self.worker.log.connect(self.append_log)
    #         self.worker.progress.connect(self.progress.setValue)
    #         self.worker.failed.connect(self.on_failed)
    #         self.worker.finished.connect(self.on_finished)
    #         self.setEnabled(False)
    #         self.worker.start()
    #     except Exception as e:
    #         self.on_failed(str(e))
    
    
    def on_run(self):
        try:
            exe = which_totalseg()
        except FileNotFoundError as e:
            QMessageBox.warning(self, "TotalSegmentator not found", str(e))
            return
    
        # ライセンス必須タスクの時だけ確認
        tdata = self.task_combo.currentData()
        if tdata in LICENSED_TASKS:
            ok_path = Path.home() / ".totalsegmentator" / "config.json"
            if not ok_path.exists():
                QMessageBox.warning(
                    self, "License required",
                    "This task requires a TotalSegmentator license.\n\n"
                    "Open Command Prompt and run:\n"
                    "  totalseg_set_license -l <YOUR_KEY>\n\n"
                    "Then click Run again."
                )
                return
    
        try:
            inp = self.input_edit.text().strip()
            if not inp:
                QMessageBox.warning(self, "Missing input", "Please select a DICOM folder or a NIfTI file.")
                return
            in_path = Path(inp)
            if not in_path.exists():
                QMessageBox.warning(self, "Invalid input", "Selected input path does not exist.")
                return
    
            # ROI収集
            rois = [self.roi_list.item(i).text() for i in range(self.roi_list.count())]
            if not rois and self.roi_input.text().strip():
                self._on_add_roi()
                rois = [self.roi_list.item(i).text() for i in range(self.roi_list.count())]
            if not rois:
                QMessageBox.warning(self, "No ROI", "Please add at least one ROI.")
                return
    
            # ===== バッチ分岐 =====
            if self.chk_batch.isChecked():
                if not in_path.is_dir():
                    QMessageBox.warning(self, "Batch input required", "In batch mode, please select a parent folder.")
                    return
    
                SUP = {".nii", ".nii.gz", ".nrrd", ".nhdr", ".dcm"}
                children = sorted(in_path.iterdir(), key=lambda p: p.name.lower())
                targets = []
                for p in children:
                    if p.is_dir():
                        targets.append(p)
                    elif p.is_file():
                        ext = (".nii.gz" if p.name.lower().endswith(".nii.gz") else p.suffix.lower())
                        if ext in SUP:
                            targets.append(p)
    
                if not targets:
                    QMessageBox.information(
                        self, "No cases found",
                        "No subfolders or supported files were found directly under the selected folder."
                    )
                    return
    
                # バッチ用状態
                self._batch_targets = targets
                self._batch_index = 0
                self._batch_rois = rois
    
                self.log.clear()
                self.progress.setValue(0)
                self.append_log(f"[I] Batch: {len(targets)} case(s) found under: {in_path}")
                self.setEnabled(False)
                self._run_next_in_batch()   # 1件目起動
                return
    
            # ===== 単体実行（従来） =====
            out_dir = self.out_edit.text().strip()
            if not out_dir:
                input_base = in_path.stem if in_path.is_file() else in_path.name
                out_dir = str(in_path.parent / f"{input_base}_Instant3D")
            Path(out_dir).mkdir(parents=True, exist_ok=True)
    
            job = Job(
                input_path=in_path,
                output_dir=Path(out_dir),
                roi_name=",".join(rois),
                plane=self.plane_combo.currentText(),
                reverse_slices=self.chk_reverse.isChecked(),
                flip_lr=self.chk_flip_lr.isChecked(),
                flip_ud=self.chk_flip_ud.isChecked(),
                use_cpu=self.chk_cpu.isChecked(),
                fastest=self.chk_fastest.isChecked(),
                smooth_iters=int(self.smooth_spin.value()),
                modality=self.modality_combo.currentText(),
                task=(self.task_combo.currentData() or "__auto__"),
                robust_crop=self.chk_robust.isChecked(),
                export_svg=self.chk_export_svg.isChecked(),
                export_csv=self.chk_export_csv.isChecked(),
            )
    
            self.log.clear()
            self.progress.setValue(0)
            self.append_log("[I] Starting…")
    
            self.worker = Worker(job, selected_rois=rois)
            self.worker.log.connect(self.append_log)
            self.worker.progress.connect(self.progress.setValue)
            self.worker.failed.connect(self.on_failed)
            self.worker.finished.connect(self.on_finished)
            self.setEnabled(False)
            self.worker.start()
    
        except Exception as e:
            self.on_failed(str(e))


    
    def _run_next_in_batch(self):
        # すべて処理済み？
        if self._batch_index >= len(self._batch_targets):
            self.setEnabled(True)
            self.append_log("[OK] Batch finished.")
            self.progress.setValue(100)
            return
    
        case_path = self._batch_targets[self._batch_index]
        input_base = case_path.stem if case_path.is_file() else case_path.name
        out_dir = case_path.parent / f"{input_base}_Instant3D"
        out_dir.mkdir(parents=True, exist_ok=True)
    
        self.append_log(f"[I] [{self._batch_index+1}/{len(self._batch_targets)}] {case_path.name}")
    
        job = Job(
            input_path=case_path,
            output_dir=out_dir,
            roi_name=",".join(self._batch_rois),
            plane=self.plane_combo.currentText(),
            reverse_slices=self.chk_reverse.isChecked(),
            flip_lr=self.chk_flip_lr.isChecked(),
            flip_ud=self.chk_flip_ud.isChecked(),
            use_cpu=self.chk_cpu.isChecked(),
            fastest=self.chk_fastest.isChecked(),
            smooth_iters=int(self.smooth_spin.value()),
            modality=self.modality_combo.currentText(),
            task=(self.task_combo.currentData() or "__auto__"),
            robust_crop=self.chk_robust.isChecked(),
            export_svg=self.chk_export_svg.isChecked(),
            export_csv=self.chk_export_csv.isChecked(),
        )
    
        self.worker = Worker(job, selected_rois=self._batch_rois)
        self.worker.log.connect(self.append_log)
    
        # 個別進捗 → 全体進捗（均等配分の簡易合成）
        def _on_item_progress(pct):
            total = len(self._batch_targets)
            done = self._batch_index
            overall = int((done * 100 + pct) / total)
            self.progress.setValue(overall)
        self.worker.progress.connect(_on_item_progress)
    
        def _on_item_failed(msg):
            self.append_log(f"[E] Failed on {case_path.name}: {msg}")
            # 失敗しても次へ
            self._batch_index += 1
            self._run_next_in_batch()
    
        def _on_item_finished(nii_path, stl_path, svg_dir):
            self._batch_index += 1
            self._run_next_in_batch()
    
        self.worker.failed.connect(_on_item_failed)
        self.worker.finished.connect(_on_item_finished)
    
        # バッチ中はUI無効のまま
        self.worker.start()







    def on_failed(self, msg: str):
        self.setEnabled(True)
        self.progress.setValue(0)
        QMessageBox.critical(self, "Error", msg)
        self.append_log("[ERR] " + msg)



    # def on_finished(self, mask_path: Path, stl_path: Path, svg_dir: Path):
    #     self.setEnabled(True)
    #     self.progress.setValue(100)
    #     self.append_log(f"[OK] Done.\nMask: {mask_path}\nSTL:  {stl_path}\nSVG:  {svg_dir}")
    #     QMessageBox.information(self, "Done", f"Export completed.\n\nMask: {mask_path}\nSTL:  {stl_path}\nSVG:  {svg_dir}")

    
    # on_finished（UI 側）
    def on_finished(self, mask_path: Path, stl_path: Path, svg_dir: Path | None):
        self.setEnabled(True)
        self.progress.setValue(100)
        svg_line = f"\nSVG:  {svg_dir}" if svg_dir else "\nSVG:  (skipped)"
        self.append_log(f"[OK] Done.\nMask: {mask_path}\nSTL:  {stl_path}{svg_line}")
        QMessageBox.information(self, "Done",
            f"Export completed.\n\nMask: {mask_path}\nSTL:  {stl_path}{svg_line}")



# ----------------- main -----------------

def main():
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
