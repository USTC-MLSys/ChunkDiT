<!---
Copyright 2022 - The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# chunkDiT for Diffusers

基于 **Diffusers v0.42.1** 的视频推理 fork，为 HunyuanVideo 13B 和 CogVideoX1.5-5B 提供 chunkDiT，并提供 VAE 解码显存优化、chunk size 离线调优及测试工具。

chunkDiT 将 token 分块执行归一化、投影、attention 和 FFN，降低中间激活的同时驻留规模。每个 query chunk 仍访问完整 K/V，保留全局 dense attention。模型权重、checkpoint 格式和 Pipeline 接口保持兼容；浮点执行顺序变化可能产生小幅数值差异。

## 支持范围

| 模型 | Pipeline | 权重 |
|---|---|---|
| HunyuanVideo 原版 13B T2V | `HunyuanVideoPipeline` | [hunyuanvideo-community/HunyuanVideo](https://huggingface.co/hunyuanvideo-community/HunyuanVideo) |
| CogVideoX 1.5-5B T2V | `CogVideoXPipeline` | [zai-org/CogVideoX1.5-5B](https://huggingface.co/zai-org/CogVideoX1.5-5B) |

当前实现面向单 GPU、BF16、`eval()` 和关闭梯度的推理。已经验证 model CPU offload、默认 VAE tiling 和 CFG 有效 batch。

## 安装

需要 Python **3.11+**、PyTorch **2.6+**，并先安装与驱动匹配的 CUDA 版 PyTorch，安装方式见 [PyTorch 官方说明](https://pytorch.org/get-started/locally/)。

在本仓库根目录执行：

```bash
python3.11 -m venv .venv
source .venv/bin/activate
# 在当前虚拟环境中安装 CUDA 版 PyTorch 后：
python -m pip install -e . \
  transformers accelerate \
  sentencepiece protobuf tiktoken imageio imageio-ffmpeg
python -c "import diffusers; print(diffusers.__version__, diffusers.__file__)"
```

最后一条命令应指向本 fork 的 `src/diffusers`。扩展接口来自本仓库，单独安装 PyPI 的原生 `diffusers` 不会获得这些接口。

## 快速开始

### HunyuanVideo 13B

```python
import torch
from diffusers import HunyuanVideoPipeline
from diffusers.utils import export_to_video

pipe = HunyuanVideoPipeline.from_pretrained(
    "hunyuanvideo-community/HunyuanVideo",
    torch_dtype=torch.bfloat16,
)
pipe.transformer.eval()
pipe.transformer.enable_chunk_dit(chunk_size=2048)
pipe.vae.enable_tiling()
pipe.vae.enable_memory_efficient_decode(conv_chunk_size=4)
pipe.enable_model_cpu_offload()

frames = pipe(
    "A cat walks on the grass, realistic style.",
    height=720,
    width=1280,
    num_frames=129,
    num_inference_steps=50,
    generator=torch.Generator("cuda").manual_seed(42),
).frames[0]
export_to_video(frames, "hunyuan.mp4", fps=24)
```

### CogVideoX 1.5-5B

```python
import torch
from diffusers import CogVideoXPipeline
from diffusers.utils import export_to_video

pipe = CogVideoXPipeline.from_pretrained(
    "zai-org/CogVideoX1.5-5B",
    torch_dtype=torch.bfloat16,
)
pipe.transformer.eval()
pipe.transformer.enable_chunk_dit(chunk_size=2048)
pipe.vae.enable_tiling()
pipe.vae.enable_memory_efficient_decode(conv_chunk_size=4)
pipe.enable_model_cpu_offload()

frames = pipe(
    "A cat walks on the grass, realistic style.",
    height=768,
    width=1360,
    num_frames=81,
    num_inference_steps=50,
    guidance_scale=6.0,
    generator=torch.Generator("cuda").manual_seed(42),
).frames[0]
export_to_video(frames, "cogvideo.mp4", fps=16)
```

已有本地 Diffusers 格式快照时，可以将模型 ID 换成本地目录。首次检查加载和输出尺寸可将步数改为 2；少步检查不能用于评估正常采样画质。

启用 model CPU offload 时，加载后直接调用 `enable_model_cpu_offload()`，保持示例中的调用顺序。手动常驻 GPU 可改用 `pipe.to("cuda")`，并根据全部驻留组件重新评估显存预算。

## 开关与 chunk size

```python
pipe.transformer.enable_chunk_dit(chunk_size=4096)
pipe.transformer.disable_chunk_dit()
```

`chunk_size` 表示 patch embedding 后每次处理的 token 上限，不是帧数或像素尺寸；必须为正整数。支持尾块、不同 batch 和文本有效长度。开启后也会分块执行最终 norm/projection，并在末层 attention 完成后释放 K/V。直接调用 Transformer 时需使用 `torch.no_grad()` 或 `torch.inference_mode()`；Pipeline 已按推理模式执行。

该开关不写入 `save_pretrained()` 的配置，重新加载后需再次启用。关闭后恢复原生执行路径。尺寸越小通常激活越低，但会增加调度开销；硬件和张量形状会影响实际延迟，宜通过下面的离线工具选择。

## 在预算内自动选择 chunk size

先准备包含 `transformer/` 子目录的模型快照。调优只加载 Transformer 的真实 BF16 权重，使用目标形状的合成输入；不运行文本编码器、scheduler 或 VAE。

```bash
CUDA_VISIBLE_DEVICES="<GPU_ID>" PYTHONPATH=src \
  python examples/chunk_dit/select_chunk_size.py \
  --model hunyuan --model-path "<HUNYUAN_MODEL_PATH>" \
  --memory-budget-gib 28 --reserve-gib 1 \
  --min-chunk-size 1024 --max-chunk-size 8192 --alignment 256 \
  --output results/hunyuan-chunk.json
```

将 `<GPU_ID>` 和 `<HUNYUAN_MODEL_PATH>` 替换为你的 GPU 编号和本地模型目录。CogVideoX 对应参数为 `--model cogvideo --model-path "<COGVIDEO_MODEL_PATH>"`；例如可使用 `--memory-budget-gib 16 --reserve-gib 1`。预算按 **GiB 的 DiT max allocated 总峰值（包含权重）**计算，预留量从预算扣除。上述 HunyuanVideo 命令的实际筛选上限为 27 GiB。

**默认只做单 block profiling，不执行完整 Transformer forward。** 通常先采样三个尺寸校准显存模型，再枚举经验可行区间中的对齐候选。每个尺寸默认测一个代表双流 block 和一个单流 block（HunyuanVideo），或一个代表 block（CogVideoX）；各预热一次、有效测量三次。按实际层数加权估计 DiT 延迟，再在满足 block 峰值预算的候选中选择最快项。α 和完整输入/输出缓冲等固定开销从样本拟合，不直接套用论文系数。

输出 JSON 的主要字段：

| 字段 | 含义 |
|---|---|
| `chunk_size` | 推荐的 token 分块尺寸 |
| `selected_block_peak_allocated` | 代表 block 的实测最高 allocated，单位字节 |
| `estimated_dit_seconds` | 按层数加权的 DiT 延迟估计 |
| `full_forward_validated` | 默认 `false`，表示未做完整 forward 核验 |
| `validations` | 默认空数组；开启完整校验后记录实际测量 |

脚本会打印启用调用，也可以手动读取：

```python
import json
with open("results/hunyuan-chunk.json") as f:
    tuning = json.load(f)
pipe.transformer.enable_chunk_dit(tuning["chunk_size"])
```

换 GPU、模型、视频形状、文本长度或有效 batch 后应重新调优。缓存签名会区分这些配置、预算、运行环境、代码和是否开启完整校验；相同成功结果会直接复用。使用 `--force` 可强制重测。

调优和耗时测试的 JSON 只保留模型类型、工作负载、预算、采样设置及测量结果。本地模型路径、权重文件信息、硬件与软件环境只参与缓存哈希计算；启动命令也仅保存哈希，不明文写入 JSON。

### 可选：完整 forward 核验

在调优命令中加上：

```bash
--validate-forward --validation-repeats 1
```

此时按 block 估算的排名，依次尝试候选的完整 Transformer forward，默认预热一次、测量一次。实际显存超限时尝试下一候选，成功后记录 `full_forward_validated=true`、`verified_peak_allocated` 和 `verified_dit_seconds`。默认模式不会生成这些 `verified_*` 字段，也不会声称完整 DiT 已通过验证。

调优在满足显存预算的候选中按估算延迟排序，不设置延迟上限。`--validation-repeats` 只在开启完整校验时生效。无可行候选会输出 `status: infeasible` 并非零退出。

`--batch-size` 指实际进入 DiT 的有效 batch，默认 HunyuanVideo 为 1、CogVideoX CFG 为 2。`--height`、`--width`、`--frames` 和 `--text-tokens` 可调整负载。默认候选范围为 1024～16384、对齐 256；上例显式限制到 8192。

block 峰值与加权延迟用于离线选择，不包含完整输入 embedding/输出头的所有开销，也不覆盖调用方持有的额外张量、文本编码器或 VAE 的峰值。新形状或紧预算可通过显式开启完整校验进一步确认。

## VAE 解码显存优化

```python
pipe.vae.enable_tiling()
pipe.vae.enable_memory_efficient_decode(conv_chunk_size=4)
# 恢复原始卷积路径：
pipe.vae.disable_memory_efficient_decode()
```

该优化保持默认 tile、overlap、GroupNorm 范围和因果缓存规则，仅对大 decoder Conv3d 按输出时间分块以降低工作区显存。`conv_chunk_size=4` 和 DiT 的 token chunk size 相互独立；支持 model CPU offload，不支持卷积上的逐层 offload hooks。该开关同样不写入 checkpoint。

实测中，decoder max allocated 约为 HunyuanVideo **7.55 GiB**、CogVideoX **8.76 GiB**。完整视频成对解码对照中，输出逐元素相同，HunyuanVideo 解码约增加 5.6% 耗时，CogVideoX 基本持平；详细条件见验证记录。

## 测试和复现

```bash
python -m pip install pytest
DIFFUSERS_TEST_DEVICE=cpu python -m pytest tests/chunk_dit -q
# CUDA 测试在可用 GPU 上执行：
CUDA_VISIBLE_DEVICES="<GPU_ID>" python -m pytest tests/chunk_dit -q
```

| 脚本 | 用途 |
|---|---|
| [select_chunk_size.py](examples/chunk_dit/select_chunk_size.py) | 在预算内通过单 block profiling 选择 chunk，完整校验默认关闭 |
| [benchmark.py](examples/chunk_dit/benchmark.py) | Pipeline 数值与分阶段显存对照，可设置少量去噪步 |
| [benchmark_dit_latency.py](examples/chunk_dit/benchmark_dit_latency.py) | 同 GPU 串行比较原生与 chunkDiT 的 DiT 延迟 |
| [benchmark_chunk_selection.py](examples/chunk_dit/benchmark_chunk_selection.py) | 测调优总耗时、block 搜索、可选完整校验和缓存耗时 |

调优耗时脚本同样默认关闭完整校验，需要时显式传入 `--validate-forward`。

## 许可证

代码采用 [Apache License 2.0](LICENSE)。模型权重的使用遵循对应模型仓库的许可。
