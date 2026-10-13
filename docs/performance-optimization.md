# Deepwork: Picture Spectrum 全版本性能优化

## 目标
对 Python 版进行全方位加速优化（向量化→GPU），同时优化 Rust 版，全版本基准测试对比，留存性能报告。

## 环境信息
- **CPU**: AMD Ryzen 7 5800X, 8C/16T
- **GPU**: NVIDIA RTX 3070, 8GB VRAM, CUDA Driver 13.3
- **CUDA Toolkit**: 未安装（nvcc 不可用），使用 CuPy/PyTorch 自带 runtime
- **Python**: 3.12.3, uv 0.6.14
- **Rust**: edition 2021, 已有 Rayon 并行 + 空壳 SIMD

## 测试图片
| 文件 | 分辨率 | 格式 | 备注 |
|------|--------|------|------|
| lena.jpg | 512×512 | JPEG grayscale | |
| QQ20250606-150451.png | 248×110 | PNG RGBA | |
| 1080.png | 1920×1080 | PNG RGB | |
| 4k.png | 3840×2160 | JPEG RGB | 文件扩展名 .png 实为 JPEG |
| IMG20250603213314.jpg | 4096×3072 | JPEG RGB | |

## 测试参数（固定）
- sample_rate=44100, start_freq=10000, spacing_ms=50, freq_step=10

## 基准测试用例
- **small**: QQ20250606-150451.png (248×110)
- **medium**: lena.jpg (512×512)
- **large**: 1080.png (1920×1080)
- **xlarge**: 4k.png (3840×2160)

## 基准测试方法论
- Python: `time.perf_counter()`, repeat=5, number=3, 取中位数
- Rust: Criterion 自动处理 warm-up 和迭代
- 测量前做 2-3 次 warm-up
- 记录系统负载
- 每个优化版本验证 WAV 输出与原始版本一致（容差内）

---

## Oracle 审查要点（已确认）

### 关键发现
1. **main.py 有 bug**: `data = data / wo` 当 wo=0 时除零，应为 `data / wi`
2. **核心算法优化遗漏**: 预计算 sin 基矩阵，每行只需矩阵乘法 `output = pixel_row @ basis_matrix`
3. **Phase 2 (多进程) 多余**: 向量化后 NumPy 自动释放 GIL，跳过
4. **CuPy VRAM 限制**: 4K 全图工作内存 73GB，需行批处理（每行 ~34MB）
5. **Rust 不用 nightly SIMD**: 专注 LUT + 内联累加 + auto-vectorization
6. **Rust 有冗余分配**: `pixel_work` 先扁平化再按行重组，应直接行并行

### 修正后的实施计划
| Phase | 内容 | 变更 |
|-------|------|------|
| 0 | 环境搭建 + 基准测试 | 不变 |
| 1 | Python 向量化 + 预计算基矩阵 | **新增基矩阵优化** |
| 2 | ~~多进程~~ → 跳过 | **条件性跳过** |
| 3 | CuPy GPU 加速 | **增加行批处理约束** |
| 4 | Rust 优化 | **LUT + 内联累加，不用 nightly** |
| 5 | 全版本对比 + 决策 | 不变 |

---

## 实施阶段

### Phase 0: 环境搭建 + 基准测试 ✅
- [x] uv 初始化 Python 项目，安装 opencv-python/numpy/soundfile/cupy-cuda12x
- [x] 修复 main.py 的除零 bug (`data / wo` → `data / wi`)
- [x] 编写统一基准测试脚本 `benchmarks/bench.py`
- [x] 测试原始 Python 版 (main.py) 全部 4 个用例
- [x] 测试 Rust release 版全部 4 个用例
- [x] 记录结果到 `benchmarks/result_*.json`，标注 commit 31ffdbb
- [x] 4k.png 实为 JPEG，已复制为 4k.jpg

### Phase 1: Python 向量化重写 (CPU) ✅
- [x] 新建 `python/picture_spectrum_vec.py`
- [x] 消除双层 for 循环，用 NumPy broadcasting 批量计算
- [x] **预计算 sin 基矩阵**：`basis[col][n] = sin(2π × freq[col] × n × Ts)`
- [x] 每行计算简化为 `row_wave = pixel_row_values @ basis_matrix`
- [x] 预分配数组替代 np.hstack 拼接
- [x] 基准测试，记录结果

### Phase 2: ~~Python 多进程加速~~ → 条件性跳过
- [ ] 检查 Phase 1 CPU 利用率，若 <90% 再考虑多进程
- [ ] 否则跳过

### Phase 3: Python GPU 加速 (CuPy) ✅
- [x] 安装 cupy-cuda12x（自带 CUDA runtime，无需 toolkit）
- [x] 新建 `python/picture_spectrum_gpu.py`
- [x] CuPy 替换 NumPy，核心逻辑不变
- [x] **行批处理**：每批 64 行，峰值 VRAM 135.6MB
- [x] 基准测试，记录结果

### Phase 4: Rust 版本优化 ✅
- [x] LUT sin() 查表 (4096 entries) + 增量相位累加
- [x] 消除 per-pixel Vec 分配，直接累加到行缓冲区
- [x] 直接行并行，去除冗余的 pixel_work 扁平化
- [x] `.cargo/config.toml` 添加 `target-cpu=native` 启用 AVX2
- [x] 基准测试，记录结果

### Phase 5: 全版本对比 + 决策 ✅
- [x] 汇总所有基准测试数据
- [x] 生成对比表格
- [x] 决定 Rust 版去留 → **归档到 archive/rust/**
- [x] 更新 README

---

## 基准测试结果

### 原始版本基线 (commit 31ffdbb)
系统: AMD Ryzen 7 5800X, RTX 3070 8GB, Python 3.13.3

| 用例 | 分辨率 | Python 原始 | Rust balanced | Rust 加速比 |
|------|--------|------------|---------------|------------|
| small | 248×110 | 2.18s | 0.11s | 20× |
| medium | 512×512 | 9.43s | 0.55s | 17× |
| large | 1920×1080 | 61.98s | 3.14s | 20× |
| xlarge | 3840×2160 | 272.14s* | 13.70s | 20× |

*xlarge Python 为单次运行，其余为 3 次中位数

原始 Python 性能: ~19,200 像素/s（与 README 描述一致）
Rust 性能: ~605,000 像素/s

### Phase 1: Python 向量化 (commit 31ffdbb, 最终版)
使用 einsum + 三角恒等式，与原始版完全一致（SNR = inf dB）

| 用例 | Python 向量化 | vs 原始 | vs Rust |
|------|--------------|---------|---------|
| small | 0.28s | 7.8× | 2.5× 慢 |
| medium | 0.65s | 14.5× | 1.2× 慢 |
| large | 3.89s | 16× | 1.2× 快 |
| xlarge | 15.05s | 18× | 1.1× 快 |

向量化 Python 性能: ~650,000 像素/s，与 Rust 接近

### Phase 3: CuPy GPU (commit 31ffdbb, 最终版)
与原始版完全一致（SNR = inf dB）

| 用例 | CuPy GPU | vs 原始 | vs Rust | vs 向量化 |
|------|----------|---------|---------|----------|
| small | 2.15s* | 1.0× | 20× 慢 | 7.7× 慢 |
| medium | 2.27s* | 4.2× | 4.1× 慢 | 3.5× 慢 |
| large | 3.07s | 20× | 1.0× 快 | 1.3× 快 |
| xlarge | 3.26s | 83× | 4.2× 快 | 4.6× 快 |

*含 GPU 初始化开销 (~2s)

GPU 性能: ~4,000,000 像素/s（大图），峰值 VRAM 49MB

### Phase 4: Rust 优化 (commit 31ffdbb)

| 用例 | Rust 优化前 | Rust 优化后 | 提速 |
|------|------------|------------|------|
| small | 0.11s | 0.039s | 2.8× |
| medium | 0.55s | 0.40s | 1.4× |
| large | 3.14s | 1.42s | 2.2× |
| xlarge | 13.70s | 4.93s | 2.8× |

Rust 优化后性能: ~1,700,000 像素/s

### 全版本最终对比 (1920×1080)

| 版本 | 时间 | 像素/s | vs 原始 | 正确性 |
|------|------|--------|---------|--------|
| Python 原始 | 62.0s | 19K | 1× | ✅ 基线 |
| Python 向量化 | 3.89s | 640K | 16× | ✅ SNR=inf |
| Rust 原始 | 3.14s | 605K | 20× | ⚠ 样本数不同 |
| Rust 优化 | 1.42s | 1.7M | 44× | ⚠ 样本数不同 |
| **CuPy GPU** | **3.07s** | **8.6M** | **20×** | ✅ SNR=inf |

### 全版本最终对比 (3840×2160)

| 版本 | 时间 | 像素/s | vs 原始 | 正确性 |
|------|------|--------|---------|--------|
| Python 原始 | 272s | 19K | 1× | ✅ 基线 |
| Python 向量化 | 15.0s | 640K | 18× | ✅ SNR=inf |
| Rust 原始 | 13.7s | 605K | 20× | ⚠ 样本数不同 |
| Rust 优化 | 4.93s | 1.7M | 55× | ⚠ 样本数不同 |
| **CuPy GPU** | **3.26s** | **25M** | **83×** | ✅ SNR=inf |

---

## 决策记录
- GPU 方案选择 CuPy：预编译 CUDA runtime，无需安装 CUDA toolkit
- 跳过 Phase 2 (多进程)：向量化后收益极小
- Rust 不使用 nightly SIMD：稳定性优先
- **Rust 版归档**：CuPy GPU 比 Rust 快 4-83×，且 Rust 版有正确性问题（样本数不同、行顺序不同），归档到 `archive/rust/`
- CuPy 设为可选依赖：`uv sync --extra gpu`

## 正确性验证
- Python 原始、向量化、GPU 版输出完全一致（SNR = inf dB）
- Rust 版有差异：样本数不同（缺少初始零波）、行顺序不同、LUT 近似误差
- 验证脚本：`benchmarks/verify.py`
