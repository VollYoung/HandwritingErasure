"""
单张图像笔迹擦除推理：加载训练好的生成器，输入带笔迹图，输出擦除结果。
"""
import argparse
from pathlib import Path

import torch
import numpy as np
from PIL import Image

from models import UnetGenerator


def tensor_to_image(tensor, to_uint8=True):
    """CWH, [-1,1] -> HWC [0,255]。"""
    arr = tensor.detach().cpu().permute(1, 2, 0).numpy()
    arr = (arr + 1.0) * 0.5
    arr = np.clip(arr, 0, 1)
    if to_uint8:
        arr = (arr * 255).astype(np.uint8)
    return arr


def main():
    parser = argparse.ArgumentParser(description="Pix2Pix 笔迹擦除推理")
    parser.add_argument("--checkpoint", type=str, required=True, help="生成器权重路径，如 checkpoints/best_generator.pth")
    parser.add_argument("--input", type=str, required=True, help="带笔迹的输入图像路径")
    parser.add_argument("--output", type=str, default="result.png", help="输出图像路径")
    parser.add_argument("--img_size", type=int, default=256, help="推理时 resize 尺寸，需与训练一致")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    net_g = UnetGenerator(3, 3, ngf=64).to(device)
    state = torch.load(args.checkpoint, map_location=device)
    net_g.load_state_dict(state)
    net_g.eval()

    img = Image.open(args.input).convert("RGB")
    img = np.array(img)
    h, w = img.shape[:2]

    import cv2
    img_resized = cv2.resize(img, (args.img_size, args.img_size), interpolation=cv2.INTER_LINEAR)
    x = img_resized.astype(np.float32) / 255.0 * 2.0 - 1.0
    x = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).float().to(device)

    with torch.no_grad():
        out = net_g(x)

    out_img = tensor_to_image(out[0])
    out_img = np.array(Image.fromarray(out_img).resize((w, h), Image.Resampling.LANCZOS))
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(out_img).save(args.output)
    print("已保存:", args.output)


if __name__ == "__main__":
    main()
