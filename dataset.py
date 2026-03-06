"""
笔迹擦除数据集：支持合成数据（在干净图上画笔迹）和成对数据（input/target 文件夹）。
"""
import os
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image


def _to_tensor(img, normalize=True):
    """HWC [0,255] -> CWH [−1,1] 或 [0,1]。"""
    arr = np.array(img)
    if len(arr.shape) == 2:
        arr = np.stack([arr] * 3, axis=-1)
    arr = arr.astype(np.float32) / 255.0
    if normalize:
        arr = arr * 2.0 - 1.0
    return torch.from_numpy(arr).permute(2, 0, 1).float()


def _draw_synthetic_handwriting(canvas, color=(0, 0, 0), thickness_range=(1, 3), num_strokes_range=(5, 25)):
    """在 canvas (numpy BGR) 上画模拟笔迹：随机折线、曲线。"""
    h, w = canvas.shape[:2]
    thickness = random.randint(*thickness_range)
    num_strokes = random.randint(*num_strokes_range)
    for _ in range(num_strokes):
        n_pts = random.randint(3, 12)
        pts = []
        for _ in range(n_pts):
            x = random.randint(0, w - 1)
            y = random.randint(0, h - 1)
            pts.append([x, y])
        pts = np.array(pts, dtype=np.int32)
        if random.random() > 0.3:
            cv2.polylines(canvas, [pts], isClosed=False, color=color, thickness=thickness, lineType=cv2.LINE_AA)
        else:
            for i in range(len(pts) - 1):
                cv2.line(canvas, tuple(pts[i]), tuple(pts[i + 1]), color=color, thickness=thickness)
    return canvas


class HandwritingErasureDataset(Dataset):
    """带笔迹图 -> 干净图。"""

    def __init__(self, data_dir, mode="synthetic", img_size=256, normalize=True):
        """
        mode: 'synthetic' 从 data_dir/clean 读图并合成笔迹；
              'paired' 从 data_dir/input 和 data_dir/target 读成对图。
        """
        self.data_dir = Path(data_dir)
        self.mode = mode
        self.img_size = img_size
        self.normalize = normalize

        if mode == "synthetic":
            clean_dir = self.data_dir / "clean"
            if not clean_dir.exists():
                raise FileNotFoundError(f"合成模式需要目录: {clean_dir}")
            self.image_paths = sorted(
                [p for p in clean_dir.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp")]
            )
            if not self.image_paths:
                raise FileNotFoundError(f"在 {clean_dir} 下未找到图片")
        else:
            in_dir = self.data_dir / "input"
            tgt_dir = self.data_dir / "target"
            if not in_dir.exists() or not tgt_dir.exists():
                raise FileNotFoundError(f"成对模式需要目录: {in_dir} 和 {tgt_dir}")
            self.image_paths = sorted(
                [p for p in in_dir.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp")]
            )
            self.input_dir = in_dir
            self.target_dir = tgt_dir
            if not self.image_paths:
                raise FileNotFoundError(f"在 {in_dir} 下未找到图片")

    def __len__(self):
        return len(self.image_paths)

    def _load_resize(self, path):
        img = Image.open(path).convert("RGB")
        img = np.array(img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
        return img

    def __getitem__(self, idx):
        if self.mode == "synthetic":
            clean_bgr = self._load_resize(self.image_paths[idx])
            target_bgr = clean_bgr.copy()
            # 随机深浅的笔迹颜色
            gray = random.randint(20, 180)
            color = (gray, gray, gray)
            input_bgr = _draw_synthetic_handwriting(clean_bgr.copy(), color=color)
        else:
            in_path = self.image_paths[idx]
            tgt_path = self.target_dir / in_path.name
            if not tgt_path.exists():
                tgt_path = self.target_dir / (in_path.stem + ".png")
            input_bgr = self._load_resize(in_path)
            target_bgr = self._load_resize(tgt_path) if tgt_path.exists() else input_bgr.copy()

        input_rgb = cv2.cvtColor(input_bgr, cv2.COLOR_BGR2RGB)
        target_rgb = cv2.cvtColor(target_bgr, cv2.COLOR_BGR2RGB)

        x = _to_tensor(input_rgb, self.normalize)
        y = _to_tensor(target_rgb, self.normalize)
        return x, y
