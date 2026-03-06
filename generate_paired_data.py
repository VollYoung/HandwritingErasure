"""
批量生成成对训练数据：干净图 -> (带笔迹图, 干净图) 保存到 data/input 与 data/target。
支持从已有 data/clean 扩展，或从零生成文档背景 + 多种笔迹样式，便于提升训练效果。
"""
import argparse
import random
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm


# ---------- 文档背景生成 ----------
def make_lined_paper(w, h, line_spacing=28, margin=25, bg_rgb=(252, 252, 250), line_rgb=(220, 218, 215)):
    """横线纸."""
    img = np.ones((h, w, 3), dtype=np.uint8)
    img[:] = bg_rgb
    for y in range(margin, h - margin, line_spacing):
        cv2.line(img, (margin, y), (w - margin, y), line_rgb, 1, lineType=cv2.LINE_AA)
    return img


def make_grid_paper(w, h, grid_size=30, margin=20, bg_rgb=(250, 251, 248), line_rgb=(230, 228, 225)):
    """方格/网格纸."""
    img = np.ones((h, w, 3), dtype=np.uint8)
    img[:] = bg_rgb
    for x in range(margin, w - margin, grid_size):
        cv2.line(img, (x, margin), (x, h - margin), line_rgb, 1, lineType=cv2.LINE_AA)
    for y in range(margin, h - margin, grid_size):
        cv2.line(img, (margin, y), (w - margin, y), line_rgb, 1, lineType=cv2.LINE_AA)
    return img


def make_plain_doc(w, h, bg_rgb=(248, 248, 246)):
    """纯色背景."""
    img = np.ones((h, w, 3), dtype=np.uint8)
    img[:] = bg_rgb
    return img


def add_print_text(img, font_scale=0.45, color=(60, 60, 60)):
    """在图上加几行印刷体文字，模拟打印文档."""
    h, w = img.shape[:2]
    try:
        font = cv2.FONT_HERSHEY_SIMPLEX
    except Exception:
        font = cv2.FONT_HERSHEY_SIMPLEX
    lines = [
        "Document Title Here",
        "Paragraph one. Some body text for layout.",
        "Paragraph two. More content to simulate print.",
    ]
    y = 30
    for line in lines:
        (tw, th), _ = cv2.getTextSize(line, font, font_scale, 1)
        cv2.putText(img, line, (25, y), font, font_scale, color, 1, cv2.LINE_AA)
        y += int(th * 1.5)
    return img


# ---------- 笔迹绘制（多种样式） ----------
def bezier_curve(p0, p1, p2, p3, steps=20):
    """四点三次贝塞尔曲线采样."""
    pts = []
    for t in np.linspace(0, 1, steps):
        u = 1 - t
        x = u * u * u * p0[0] + 3 * u * u * t * p1[0] + 3 * u * t * t * p2[0] + t * t * t * p3[0]
        y = u * u * u * p0[1] + 3 * u * u * t * p1[1] + 3 * u * t * t * p2[1] + t * t * t * p3[1]
        pts.append([int(x), int(y)])
    return np.array(pts, dtype=np.int32)


def draw_cursive_stroke(canvas, color, thickness, h, w):
    """模拟一笔连写：多段贝塞尔曲线."""
    n_segments = random.randint(2, 5)
    pts = []
    x, y = random.randint(w // 4, 3 * w // 4), random.randint(h // 4, 3 * h // 4)
    for _ in range(n_segments):
        p0 = (x, y)
        p1 = (x + random.randint(-40, 40), y + random.randint(-30, 30))
        p2 = (p1[0] + random.randint(-40, 40), p1[1] + random.randint(-30, 30))
        p3 = (p2[0] + random.randint(-30, 30), p2[1] + random.randint(-20, 20))
        p3 = (np.clip(p3[0], 10, w - 11), np.clip(p3[1], 10, h - 11))
        curve = bezier_curve(p0, p1, p2, p3)
        pts.extend(curve)
        x, y = p3[0], p3[1]
    if len(pts) >= 2:
        pts = np.array(pts, dtype=np.int32)
        cv2.polylines(canvas, [pts], False, color, thickness, lineType=cv2.LINE_AA)


def draw_polyline_stroke(canvas, color, thickness, h, w):
    """随机折线（原逻辑增强）.."""
    n_pts = random.randint(4, 14)
    pts = []
    for _ in range(n_pts):
        pts.append([random.randint(15, w - 16), random.randint(15, h - 16)])
    pts = np.array(pts, dtype=np.int32)
    if random.random() > 0.25:
        cv2.polylines(canvas, [pts], False, color, thickness, lineType=cv2.LINE_AA)
    else:
        for i in range(len(pts) - 1):
            cv2.line(canvas, tuple(pts[i]), tuple(pts[i + 1]), color, thickness, lineType=cv2.LINE_AA)


def draw_underline(canvas, color, thickness, h, w):
    """横线/下划线."""
    y = random.randint(40, h - 40)
    x1 = random.randint(20, w // 2)
    x2 = random.randint(w // 2, w - 21)
    cv2.line(canvas, (x1, y), (x2, y), color, thickness, lineType=cv2.LINE_AA)


def draw_margin_note(canvas, color, thickness, h, w):
    """侧边/页边批注：短竖线+小折线."""
    x = random.choice([random.randint(15, 45), random.randint(w - 46, w - 16)])
    y1 = random.randint(30, h - 80)
    y2 = y1 + random.randint(30, 60)
    cv2.line(canvas, (x, y1), (x, y2), color, thickness, lineType=cv2.LINE_AA)
    for _ in range(random.randint(1, 3)):
        x2 = x + random.randint(10, 40) if x < w // 2 else x - random.randint(10, 40)
        x2 = np.clip(x2, 10, w - 11)
        cv2.line(canvas, (x, y1), (x2, y1 + 5), color, max(1, thickness - 1), lineType=cv2.LINE_AA)
        y1 += 12


def draw_scribble_block(canvas, color, thickness, h, w):
    """一小块涂鸦区域：密集短线段."""
    cx = random.randint(60, w - 61)
    cy = random.randint(60, h - 61)
    for _ in range(random.randint(8, 20)):
        dx = random.randint(-25, 25)
        dy = random.randint(-25, 25)
        cv2.line(canvas, (cx, cy), (cx + dx, cy + dy), color, thickness, lineType=cv2.LINE_AA)
        cx, cy = cx + dx, cy + dy
        cx = np.clip(cx, 10, w - 11)
        cy = np.clip(cy, 10, h - 11)


STROKE_DRAWERS = [
    draw_cursive_stroke,
    draw_polyline_stroke,
    draw_underline,
    draw_margin_note,
    draw_scribble_block,
]


def draw_synthetic_handwriting_diverse(canvas_bgr, intensity="medium", num_strokes_range=(6, 28)):
    """
    在 canvas 上画多种笔迹，尽量贴近真实手写。
    intensity: 'light' / 'medium' / 'heavy'
    """
    h, w = canvas_bgr.shape[:2]
    if intensity == "light":
        thickness_range = (1, 2)
        num_strokes = random.randint(*num_strokes_range) // 2
        gray_range = (80, 140)
    elif intensity == "heavy":
        thickness_range = (2, 4)
        num_strokes = random.randint(num_strokes_range[0] + 4, num_strokes_range[1] + 10)
        gray_range = (20, 100)
    else:
        thickness_range = (1, 3)
        num_strokes = random.randint(*num_strokes_range)
        gray_range = (30, 160)

    for _ in range(num_strokes):
        gray = random.randint(*gray_range)
        color = (gray, gray, gray)
        thickness = random.randint(*thickness_range)
        drawer = random.choice(STROKE_DRAWERS)
        try:
            drawer(canvas_bgr, color, thickness, h, w)
        except Exception:
            draw_polyline_stroke(canvas_bgr, color, thickness, h, w)
    return canvas_bgr


# ---------- 主流程 ----------
def generate_clean_backgrounds(out_dir: Path, count: int, size: int):
    """生成 count 张干净背景到 out_dir."""
    out_dir.mkdir(parents=True, exist_ok=True)
    generators = [
        ("lined", lambda: make_lined_paper(size, size)),
        ("grid", lambda: make_grid_paper(size, size)),
        ("plain", lambda: make_plain_doc(size, size)),
    ]
    for i in tqdm(range(count), desc="生成干净背景"):
        name, gen = generators[i % len(generators)]
        img = gen()
        if random.random() > 0.3:
            img = add_print_text(img)
        path = out_dir / f"doc_{i:04d}_{name}.png"
        cv2.imwrite(str(path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


def load_clean_images(clean_dir: Path, size: int):
    """从已有 clean 目录加载图片并 resize."""
    paths = sorted(
        p for p in clean_dir.iterdir()
        if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp")
    )
    imgs = []
    for p in paths:
        img = cv2.imread(str(p))
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)
        imgs.append((p.stem, img))
    return imgs


def main():
    parser = argparse.ArgumentParser(description="批量生成成对训练数据 (input=带笔迹, target=干净)")
    parser.add_argument("--data_dir", type=str, default="data", help="数据根目录")
    parser.add_argument("--clean_dir", type=str, default=None, help="干净图目录，默认 data/clean；若不存在则自动生成背景")
    parser.add_argument("--size", type=int, default=256, help="输出图像尺寸")
    parser.add_argument("--num_base", type=int, default=50, help="干净图数量（在无 clean_dir 时生成）")
    parser.add_argument("--variants", type=int, default=5, help="每张干净图生成的带笔迹变体数")
    parser.add_argument("--intensity", type=str, default="medium", choices=["light", "medium", "heavy"],
                        help="笔迹密度/深浅")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    data_dir = Path(args.data_dir)
    input_dir = data_dir / "input"
    target_dir = data_dir / "target"
    input_dir.mkdir(parents=True, exist_ok=True)
    target_dir.mkdir(parents=True, exist_ok=True)

    clean_dir = Path(args.clean_dir) if args.clean_dir else data_dir / "clean"
    if clean_dir.exists():
        pairs = load_clean_images(clean_dir, args.size)
        if not pairs:
            raise FileNotFoundError(f"在 {clean_dir} 下未找到有效图片")
        print(f"从 {clean_dir} 加载 {len(pairs)} 张干净图")
    else:
        clean_dir.mkdir(parents=True, exist_ok=True)
        generate_clean_backgrounds(clean_dir, args.num_base, args.size)
        pairs = load_clean_images(clean_dir, args.size)
        print(f"已生成 {len(pairs)} 张干净背景到 {clean_dir}")

    total = 0
    for stem, clean_rgb in tqdm(pairs, desc="生成成对数据"):
        clean_bgr = cv2.cvtColor(clean_rgb, cv2.COLOR_RGB2BGR)
        for v in range(args.variants):
            with_handwriting = draw_synthetic_handwriting_diverse(clean_bgr.copy(), intensity=args.intensity)
            name = f"{stem}_v{v:02d}.png"
            cv2.imwrite(str(input_dir / name), with_handwriting)
            cv2.imwrite(str(target_dir / name), clean_bgr)
            total += 1
    print(f"已生成 {total} 对训练数据: {input_dir} / {target_dir}")
    print("训练命令: python train.py --data_dir data --mode paired --epochs 100")


if __name__ == "__main__":
    main()
