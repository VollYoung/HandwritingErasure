"""
生成演示用合成数据：在 data/clean 下创建几张“干净文档”图，便于直接运行 train --mode synthetic。
"""
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def main():
    data_dir = Path("data/clean")
    data_dir.mkdir(parents=True, exist_ok=True)

    for i in range(5):
        w, h = 256, 256
        img = Image.new("RGB", (w, h), color=(250, 250, 248))
        draw = ImageDraw.Draw(img)
        for y in range(40, h, 30):
            draw.line([(20, y), (w - 20, y)], fill=(220, 220, 218), width=1)
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 14)
        except Exception:
            font = ImageFont.load_default()
        draw.text((30, 50), f"Document sample {i + 1}", fill=(80, 80, 80), font=font)
        img.save(data_dir / f"doc_{i:02d}.png")
    print(f"已在 {data_dir} 下生成 5 张示例图。运行: python train.py --data_dir data --mode synthetic --epochs 50")


if __name__ == "__main__":
    main()
