# Qwen3.8-27B Q8MTP artifact: what the weight formats add up to

A breakdown of the 27B acceptance artifact's tensor payload by block format, and therefore of
what the `Q2LynnStyle-Q8MTP` name is claiming. Arrived with the KVMem patch; kept because it is
the only written record of that artifact's composition.

主要权重格式如下：

| 格式                      |    约占权重 |
| ------------------------- | ----------: |
| `gguf_iq3_s`              |   3.888 GiB |
| `gguf_iq4_xs`             |   2.628 GiB |
| `gguf_q4_k`               |   2.304 GiB |
| `gguf_q5_k`               |   1.628 GiB |
| `gguf_iq3_xxs`            |   1.017 GiB |
| `gguf_iq2_s`              |   0.558 GiB |
| `gguf_q8_0`               |   0.420 GiB |
| 自定义 `q4/q5/q6_g*_fp16` | 约 0.23 GiB |
| BF16/FP32 等              |        少量 |

总 tensor payload 约 **12.82 GiB**。

所以文件名中的：

```text
Q2LynnStyle-Q8MTP
```

不能理解为“主模型全部是 Q8”。实际主模型大量使用 `IQ3_S`、`IQ4_XS`、`Q4_K` 等低比特格式。

- NInfer artifact 有 1133 个 tensor；
- NInfer 同时包含 `gguf_*` 格式、NInfer 自有 `q*_g64_fp16`、BF16、FP32、MTP 和 Vision 相关对象。

## 4. SM75 的 BF16 MMA 不是原生 BF16

Turing SM75 没有 Ampere 的 BF16 Tensor Core MMA。当前移植在 `src/ops/common/mma.cuh` 中采取了：

```text
BF16 -> FP16 -> 两次 m16n8k8 MMA
```

这会产生两类差异：

1. BF16 的指数范围比 FP16 大；
2. 当前 BF16->FP16 路径会对超出 FP16 范围的有限值进行饱和。

这比直接溢出成 `inf` 安全得多，但与真正的 BF16 MMA 不等价。它是当前移植中最重要的数值差异之一。

受影响的主要路径包括：

- BF16 线性层；
- GDN gating projection；
- 部分 attention projection；
- 其他调用 `mma_bf16()` 的路径。

因此，即使 KV 使用 BF16，**T10 上也不等价于 RTX 3090/Ampere 上的 BF16 运行**。

---

## 5. `mma_tf32_bits()` 的 Turing 模拟路径需要重点验证

SM75 没有 TF32 MMA，当前代码将 TF32 操作数转换为 FP16 后执行 FP16 MMA，并且使用 warp shuffle 重新组装 fragment。

代码中 `mma_tf32_bits()` 使用的是：

```cpp
__floats2half2_rn(...)
```

而不是带饱和的转换。

这意味着如果某个 FP32 操作数超过 FP16 可表示范围，可能转成 `inf`，进一步出现：

```text
inf * 0 -> NaN
```

这条路径主要被 GDN recurrence/state-passing 使用。当前代码在 BF16->FP16 及 recurrent state 转换处已经加入了饱和处理，但 `mma_tf32_bits()` 本身没有采用同样的保护。

这不代表当前一定会出错，因为正常模型激活可能从不超过 FP16 范围；但从代码审查角度，这是一个应当加入 T10 数值测试的风险点：

- 输入普通范围数据；
- 输入接近 FP16 上限的数据；
- 输入包含大但有限的 FP32 数据；
- 检查输出是否出现 NaN/Inf；
- 与 FP32 reference 对比。



# 四、当前 T10 移植中对速度影响最大的限制

## 1. 没有 `cp.async`

Turing 不支持 Ampere 的：

```text
cp.async
```

当前 `src/ops/common/memory.cuh` 将其替换成普通 global load + shared store，并把等待操作变成 no-op。

这保证了功能和同步语义，但失去了：

- global-to-shared 异步传输；
- load/compute 重叠；
- 多 stage pipeline 的主要收益。

因此长 prompt prefill 很可能是 T10 上最大的瓶颈之一。

可能的后续优化方向：

- warp-specialized loader；
- 使用寄存器预取和双缓冲；
- 将 global load 分散到 MMA 之间；
- 针对 T10 的 32K/64K shared memory tile 重新设计；
- 避免所有 warp 同时等待同步 copy。

这类优化不会改变模型数值，只改变 load/compute 调度，但需要实测寄存器压力和 occupancy。

---

## 2. Turing shared memory 限制迫使很多路径退化

当前代码明确把以下路径从大 tile/prompt kernel 切换到小 tile 或 chunked 路径：

- causal attention prompt route；
- INT8/rk8v4 attention；
- GDN cooperative split；
- 部分 64-key dynamic shared-memory route；
- QSA indexer TF32 MMA route。

原因是 T10 每个 SM 的 shared memory 较小，原始 Ampere route 无法驻留或直接无法 launch。

日志中还显示：

```text
sm_75 has no calibrated route profile and cannot be calibrated;
using the compiled routes
```

这意味着当前 T10 没有真正的设备校准 profile，而是在使用编译时 route。

这是一个明确的性能优化点：

- 为 SM75 单独做 route sweep；
- 给 rk8v4/int8 attention 建立 T10 专用 route table；
- 重新测试 tile width、CTA 数量、key block 大小；
- 对 prefill、decode、MTP verification 分开校准。

不能直接使用 RTX 3090 的 profile，因为 shared memory、Tensor Core shape 和 occupancy 都不同。

---

## 3. BF16 MMA 的转换开销较大

当前 `mma_bf16()` 每次 MMA 调用都可能执行：

```text
BF16 unpack
BF16 -> float
float -> FP16
两次 m16n8k8 MMA
```

相比 Ampere 原生：

```text
一次 m16n8k16 BF16 MMA
```

T10 上可考虑：

- 在 tile 装载阶段一次性把 BF16 转成 FP16；
- 共享内存中直接保存 FP16 tile；
- 后续 MMA 复用转换后的数据；
- 避免每个 MMA 调用重复转换；
- 按 kernel 的真实 outlier 范围决定是否需要饱和检查。

这可能是 SM75 上 BF16 projection/GDN 路径最值得做的 kernel 优化之一。

---

## 4. 当前 `qsa_indexer` 放弃了 TF32 MMA

SM75 路径中，`qsa_indexer` 的 TF32 WMMA kernel 被排除，改走 FP32 SIMT：

```text
Turing -> FP32 SIMT score kernel
```

这保证了可运行，但可能拖慢长上下文索引选择。

可行方向：

- 使用 Turing 原生 FP16 MMA 做近似 score；
- 对 score 做分块和 warp reduction；
- 使用 FP16/FP32 混合路径；
- 只在候选排序阶段使用低精度，最终保留 FP32 校验；
- 针对实际 capacity 测量 indexer 占比，再决定是否值得优化。

如果 QSA indexer 只占很小的总时间，不应优先于权重 GEMV 或 attention。

---