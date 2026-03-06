# Pix2Pix 笔迹擦除示例

基于 Pix2Pix (cGAN) 的文档笔迹擦除：输入带手写字的图像，输出擦除笔迹后的干净图像。

## 环境

```bash
pip install -r requirements.txt
```

## 快速开始（合成数据演示）

无现成数据时，可先生成演示用的干净文档图，再训练：

```bash
python prepare_demo_data.py   # 在 data/clean 下生成 5 张示例图
python train.py --data_dir data --mode synthetic --epochs 50
```

## 批量生成成对数据（推荐，效果更好）

用脚本一次性生成多张「带笔迹 / 干净」成对图，再按成对数据训练：

```bash
# 无 data/clean 时：自动生成 50 张背景 + 每张 5 个笔迹变体 → 共 250 对
python generate_paired_data.py --data_dir data --num_base 50 --variants 5

# 已有 data/clean 时：基于现有干净图生成多组笔迹变体
python generate_paired_data.py --data_dir data --variants 8

# 笔迹更密/更淡
python generate_paired_data.py --data_dir data --num_base 80 --variants 6 --intensity heavy
```

参数：`--num_base` 干净图数量，`--variants` 每张的变体数，`--intensity` 可选 `light`/`medium`/`heavy`。生成结果在 `data/input/` 与 `data/target/`，随后用成对模式训练：

```bash
python train.py --data_dir data --mode paired --epochs 100
```

## 数据准备

- **方式一**：使用合成数据。将干净文档图放在 `data/clean/`，训练时实时在图上叠加笔迹（随机性强，效果一般）。
- **方式二（推荐）**：成对数据。用上面脚本生成，或自行准备：`data/input/` 放带笔迹图，`data/target/` 放对应干净图，文件名一一对应。

目录结构示例：

```
data/
  clean/          # 仅合成模式：干净文档图
  或
  input/          # 带笔迹图
  target/         # 干净图（与 input 同名）
```

## 训练

```bash
# 使用合成数据（从 data/clean 生成输入-目标对）
python train.py --data_dir data --mode synthetic --epochs 100

# 使用已有成对数据
python train.py --data_dir data --mode paired --epochs 100
```

可选参数：`--batch_size`、`--lr`、`--img_size` 等，见 `train.py --help`。

## 推理

```bash
python inference.py --checkpoint checkpoints/best_generator.pth --input path/to/image_with_handwriting.png --output result.png
```

## 结构说明

- `models/generator.py`: U-Net 生成器
- `models/discriminator.py`: PatchGAN 判别器
- `dataset.py`: 数据加载与合成笔迹
- `train.py`: 训练脚本（对抗损失 + L1）
- `inference.py`: 单张图像推理
- `generate_paired_data.py`: 批量生成成对数据（多种笔迹样式 + 横线/方格/纯色背景）

## 参考

Pix2Pix: Image-to-Image Translation with Conditional Adversarial Networks (Isola et al., CVPR 2017)
