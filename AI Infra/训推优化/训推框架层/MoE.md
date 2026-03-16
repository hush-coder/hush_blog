MoE 是 Mixture of Experts（混合专家）的缩写。它是一种神经网络架构设计，核心思想是：不用一个巨大的密集模型处理所有输入，而是训练多个“专家”子网络，并设置一个“门控”机制，让每个输入只激活部分专家，从而在保持模型总容量的同时，大幅降低计算成本

# 工作原理

一个 MoE 层通常包含：

1. N 个专家网络（Experts）：每个专家通常是一个小型前馈网络（FFN）。
2. 一个门控网络（Gating Network）：通常是一个 Softmax 层，输出一个概率分布，表示每个专家对当前输入的重要性。

**前向计算过程：**

- 输入`x`同时进入门控网络和所有专家。
- 门控网络输出权重 
- 最终输出为各专家输出的加权和：

但在实际大规模部署中，为了节省计算，不会真的计算所有专家，而是只取 Top-K 个专家（比如 K=1 或 K=2），其余权重置零，这样只有少数专家被激活。

```python
class MoELayer(nn.Module):
    def __init__(self, num_experts, expert_dim):
        self.gate = nn.Linear(expert_dim, num_experts)  # 门控网络
        self.experts = nn.ModuleList([Expert() for _ in range(num_experts)]) # 专家列表

    def forward(self, x):
        # 1. 门控计算权重 (动态决策)
        gate_logits = self.gate(x)
        gate_weights = torch.softmax(gate_logits, dim=-1)

        # 2. 选择 Top-K 专家
        top_k_weights, top_k_indices = torch.topk(gate_weights, k=2)

        # 3. 路由：将 tokens 分发到选中的专家
        expert_outputs = self.route_and_compute(x, top_k_indices, top_k_weights)

        # 4. 加权求和
        output = combine_outputs(expert_outputs, top_k_weights)
        return output
```

# 如何为 MoE 编写高性能 Kernel

为 MoE 编写 Kernel，核心挑战在于处理其动态稀疏性。MoE 的专家路由结果（哪个 Token 去哪个专家）是动态的，导致数据流不规则。高性能 Kernel 的目标就是尽可能地将这种不规则转化为规则计算，以充分利用 GPU 的算力。

通常，MoE 的 Kernel 实现有以下几个关键层次：

## 1. 基础范式：Grouped GEMM（分组矩阵乘法）

这是最常用的实现方式。把给同一个专家的所有 Token 拼接在一起，然后为每个专家调用一次 GEMM。

- 朴素实现：for 循环遍历每个专家，调用 cublasGemmEx。

    - 问题：专家数量可能很大（数十到数千），串行启动 Kernel 会产生巨大的启动开销，且无法利用 GPU 的并行性。

- Grouped GEMM：NVIDIA 的 cuBLAS 和 CUTLASS 提供了 Grouped GEMM 功能。它将多个小的 GEMM 问题打包成一个 Kernel 启动。

    - 原理：Kernel 内部维护一个队列，多个 SM（流多处理器）可以动态地从队列中取任务（某个专家的 GEMM 计算）。

    - 优点：减少了 Kernel 启动开销，且能更好地平衡负载（如果一个专家计算量小，SM 可以快速处理完再去处理下一个）。

## 2. 进阶优化：Fused MoE（融合门控与分发）

Grouped GEMM 解决了计算问题，但数据准备（Dispatch）的开销依然存在。Token 需要根据门控结果被重新排列（Gather）到专家专用的连续内存中，计算完后还要再排列回来（Scatter）。

- Fused Kernel：将“Gather-计算-Scatter”融合成一个 Kernel。

- 流程：Kernel 读取原始 Token 数据，根据门控的索引，直接将数据送入对应专家的计算流水线，计算结果直接写回输出位置。

- 优点：避免了中间显存的读写（通常称为 Drop 和 Add 操作），显著降低延迟和显存占用。

- 实现：这通常需要借助 CUTLASS 这样的模板库，定制 Kernel 的主循环。

## 3. 高级优化：针对 Transformer 的极致优化

当 MoE 应用于 Transformer 时，专家通常是 FFN（前馈网络）层。可以结合 Transformer 的特性进行优化：

- Stream-K：一种 GEMM 分解方式，能将 GEMM 计算更细粒度地分解，在 SM 之间实现完美负载均衡。特别适合处理 MoE 中因各专家 Token 数不同导致的“长短不齐”问题。

- 利用 Tensor Core：确保 Grouped GEMM 能正确使用 Tensor Core（Hopper 架构的 WGMMA 指令）。这需要精心布置数据布局（如建议使用 cutlass::layout::RowMajor 配合特定指令）。

- Shared Memory 规划：在 Fused Kernel 中，合理分配 Shared Memory 用于缓存权重和输入，减少对 HBM（高带宽内存）的访问。

## 4. 分布式通信：All-to-All 优化

如果专家分布在不同的 GPU 上，就需要跨设备通信。

- All-to-All 的本质：每个 GPU 上的部分 Token 需要发送给其他 GPU（因为要去的专家在其他卡上）。这是一个典型的转置操作。

- 优化点：

    - 通信与计算重叠：发送数据的同时，对留在本地的 Token 进行计算。
    - 拓扑感知：利用 NVLink 直连带宽，避免通过 PCIe 绕路。
    - RMA（远程内存直接访问）：在 NVIDIA 的 NVSHMEM 或 InfiniBand 支持下，实现 GPU 直接 Put/Get，减少 CPU 介入。

# 如何在MLIR中体现

MLIR 的强大之处在于它有不同层次的抽象（Dialect）。MoE 的动态路由（控制流）在 MLIR 的不同 Dialect 中可以有不同形式的表示。

## 1. 高层次表示（框架视角）：mhlo 或 torch Dialect

在框架接入 MLIR 的早期阶段，MoE 可能仍然保留为高级语义操作。

***方案 A：表示为函数调用 + 控制流***

使用 scf.if（结构化控制流）或 cf.cond_br（非结构化控制流）来表示 topk 带来的分支。

将每个专家的计算封装在 func.call 中。

**缺点**：不够“显式”，后续优化 Pass 难以识别出这是一个 MoE 模式，难以做特定优化（如 Fuse 通信）。

***方案 B：自定义 mhlo.moe Op***

定义一个高级 Op：mhlo.moe，它接受输入、门控权重，并输出最终结果。

这个 Op 内部隐式包含了门控、分发、专家计算和合并的逻辑。

**优点**：在早期阶段保留了高层语义，便于后续的 MoE 专用 Pass 进行优化（如专家负载均衡、SparseCore 映射）。

## 2. 中层次表示（经过部分 Lowering 后）：linalg + scf Dialect

随着 Lowering 的进行，自定义 Op 会被逐步拆解。

- 门控（TopK）：会被 Lower 成一系列 linalg.generic 操作，可能伴随着 arith.cmpf（浮点比较）和 scf.if。
- 分发（Dispatch）：这是一个关键操作。在 MLIR 中，可以表示为：
    - tensor.gather：根据索引收集数据，生成一个新的、排好序的张量。这个张量的形状可能是动态的，需要通过 tensor.empty 和 tensor.insert_slice 配合动态维度。
    - 控制流：scf.for 循环遍历专家，循环体内是 linalg.matmul（矩阵乘法）。这对应了朴素的串行专家计算。
- 专家计算：如果用 linalg.matmul，它已经是对 GEMM 的良好抽象。

## 3. 低层次表示（接近硬件）：gpu + nvvm Dialect
最终，所有控制流和高阶操作都需要 Lower 到接近硬件的表示。

- 控制流：scf.for 和 scf.if 会被 Lower 为 cf Dialect 的基本块和分支，最终映射到 GPU 的指令级控制流。在 nvvm Dialect 中，会有对应的分支指令。

- 计算：linalg.matmul 会被转换为对 gpu.launch 的调用，内部包含 gpu.blocks 和 gpu.threads 的循环。或者，更直接地，通过 vector Dialect 和 nvvm Dialect（如 nvvm.wgmma）直接生成 Tensor Core 指令。

- 同步：gpu.barrier 会被插入以同步块内的线程。

## 4. MLIR 中处理 MoE 动态性的关键：tensor.dim 和动态类型
MoE 带来的动态形状（Dynamic Shapes）是 MLIR 类型系统必须处理的。

- tensor<*xf32>：完全未指定的张量。

- tensor<?x?xf32>：维度是动态的（用 ? 表示）。

- tensor.dim 操作：在运行时查询张量某个维度的大小。

- memref.reinterpret_cast：在缓冲区层面重新解释形状。

编译器必须能够传播这些动态维度信息，确保生成的代码能正确计算循环边界和内存偏移。

# 挑战

## 1. 动态路由与负载不均衡
门控网络的选择是数据依赖的，导致不同 batch 或 token 激活的专家不同。

这可能造成专家负载不均衡：某些专家处理大量 token，某些闲置。

对算子实现：需要支持动态稀疏的 gather 操作，将不同 token 的数据按专家分拆、合并，这会引入额外的通信和内存拷贝。

## 2. All-to-All 通信
在分布式训练中，专家可能分布在不同的 GPU 上。如果 token 需要路由到其他 GPU 上的专家，就需要进行跨设备通信（通常是 All-to-All 集体通信）。

这对通信带宽和延迟有很高要求，需要高效实现。

## 3. 内核融合难度增加
因为路由是动态的，静态图优化（如算子融合）更难应用。编译器需要支持动态形状和条件执行。

## 4. 显存管理
专家参数虽然多，但每次只加载部分专家。如何高效地在显存中换入换出专家参数（类似专家并行）是一个优化点。

