"""
Pix2Pix 笔迹擦除训练脚本：对抗损失 + L1 重建损失。
"""
import argparse
import os
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import HandwritingErasureDataset
from models import UnetGenerator, PatchGANDiscriminator


def main():
    parser = argparse.ArgumentParser(description="Pix2Pix 笔迹擦除训练")
    parser.add_argument("--data_dir", type=str, default="data", help="数据根目录")
    parser.add_argument("--mode", type=str, default="synthetic", choices=["synthetic", "paired"], help="synthetic=从 clean 合成笔迹, paired=成对 input/target")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--lambda_l1", type=float, default=100.0, help="L1 损失权重")
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    dataset = HandwritingErasureDataset(args.data_dir, mode=args.mode, img_size=args.img_size)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)

    net_g = UnetGenerator(3, 3, ngf=64).to(device)
    net_d = PatchGANDiscriminator(6, ndf=64).to(device)

    opt_g = torch.optim.Adam(net_g.parameters(), lr=args.lr, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(net_d.parameters(), lr=args.lr, betas=(0.5, 0.999))

    bce = nn.BCEWithLogitsLoss()
    l1 = nn.L1Loss()

    for epoch in range(1, args.epochs + 1):
        net_g.train()
        net_d.train()
        total_g = total_d = 0.0
        pbar = tqdm(loader, desc=f"Epoch {epoch}/{args.epochs}")

        for x, y in pbar:
            x, y = x.to(device), y.to(device)

            # 生成
            fake = net_g(x)

            # 判别器：真 (x,y) 为 1，假 (x, fake) 为 0
            opt_d.zero_grad()
            pred_real = net_d(x, y)
            pred_fake = net_d(x, fake.detach())
            real_label = torch.ones_like(pred_real, device=device)
            fake_label = torch.zeros_like(pred_fake, device=device)
            loss_d = bce(pred_real, real_label) + bce(pred_fake, fake_label)
            loss_d.backward()
            opt_d.step()
            total_d += loss_d.item()

            # 生成器：让 (x, fake) 被判为真 + L1(fake, y)
            opt_g.zero_grad()
            pred_fake_g = net_d(x, fake)
            loss_gan = bce(pred_fake_g, real_label)
            loss_l1 = l1(fake, y)
            loss_g = loss_gan + args.lambda_l1 * loss_l1
            loss_g.backward()
            opt_g.step()
            total_g += loss_g.item()

            pbar.set_postfix(loss_g=total_g / (pbar.n + 1), loss_d=total_d / (pbar.n + 1))

        n = len(loader)
        print(f"Epoch {epoch} loss_g={total_g/n:.4f} loss_d={total_d/n:.4f}")

        if epoch % 10 == 0 or epoch == args.epochs:
            torch.save(net_g.state_dict(), save_dir / f"generator_epoch{epoch}.pth")
            torch.save(net_d.state_dict(), save_dir / f"discriminator_epoch{epoch}.pth")
    torch.save(net_g.state_dict(), save_dir / "best_generator.pth")
    print("训练完成，模型已保存到", save_dir)


if __name__ == "__main__":
    main()
