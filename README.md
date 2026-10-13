# 一种基于频谱的高质量图片传输方案

> **关于本项目**：这是 [fjqz177/Picture_spectrum](https://github.com/fjqz177/Picture_spectrum)
> 的 fork，在其基础上做了全版本性能优化（Python 向量化 / CuPy GPU / Rust 实验）。
> 优化过程与决策记录见 `docs/performance-optimization.md`。

## 目录结构

```
.
├── main.py                       # 原版入口脚本（保留除零修复）
├── python/
│   ├── picture_spectrum_vec.py   # CPU 向量化实现（默认推荐，~16x）
│   ├── picture_spectrum_gpu.py   # CuPy GPU 实现（~83x，需 NVIDIA GPU）
│   └── bench_original.py         # 原始算法基线（用于对照）
├── benchmarks/
│   ├── bench.py                  # 统一基准测试脚本
│   ├── verify.py                 # 各版本输出一致性校验（SNR）
│   └── results/                  # 基准测试结果 JSON
├── docs/
│   └── performance-optimization.md  # 优化过程与决策记录
├── samples/                      # 测试图片（仅 lena.jpg 入库，其余见 .gitignore）
└── archive/rust/                 # 已归档的 Rust 实验版（正确性存疑，见其 README）
```

## 性能

| 版本 | 1080p 耗时 | 4K 耗时 | 像素/s | 正确性 |
|------|-----------|---------|--------|--------|
| Python 原始 | 62s | 272s | 19K | ✅ 基线 |
| Python 向量化 (CPU) | 3.9s | 15s | 640K | ✅ 完全一致 |
| **CuPy GPU** | **3.1s** | **3.3s** | **25M** | ✅ 完全一致 |

GPU 版比原始版本快 **83×**（4K）。

## 快速开始

```bash
# 安装 uv (如未安装)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 安装依赖 (CPU 版)
uv sync

# 安装依赖 (GPU 版，需要 NVIDIA GPU)
uv sync --extra gpu

# 运行 (CPU 向量化版)
uv run python python/picture_spectrum_vec.py --image input.jpg -o output.wav

# 运行 (GPU 版)
uv run python python/picture_spectrum_gpu.py --image input.jpg -o output.wav
```

## 参数说明

- `--image`: 输入图片路径（必需）
- `-r, --sample-rate`: 采样频率（默认 44100）
- `-f, --start-freq`: 起始频率（默认 10000）
- `-s, --spacing-ms`: 频谱行距，毫秒（默认 50）
- `-z, --freq-step`: 频谱列距，赫兹（默认 10）
- `-o, --output`: 输出文件名（可选）

## 演示视频
https://www.bilibili.com/video/BV1oh4y1k7tK/

## 高质量频谱效果 
原图分辨率 1080 x 1080  
![image](https://github.com/WOLF4096/Picture_spectrum/assets/98315254/813b5e34-88c1-45f3-955f-6876542321ec)
![image](https://github.com/WOLF4096/Picture_spectrum/assets/98315254/d6d2be00-3b3e-46d0-b826-3e4d3d9cbe4e)

## 低占用频谱效果
可用于无线电传图，比SSTV还快，可自定义带宽  
最低只需占用 横向分辨率 x 2 的带宽  
例如 160 x 120 的图只需占用 320 Hz 就能看清    
![image](https://github.com/WOLF4096/Picture_spectrum/assets/98315254/19965550-f7ff-4ce9-91af-96712287d270)

## 基准测试

详细测试数据见 `benchmarks/results/` 目录。

```bash
# 运行基准测试
uv run python benchmarks/bench.py --label test --python-script python/picture_spectrum_vec.py --images small medium large

# 验证正确性
uv run python benchmarks/verify.py
```
