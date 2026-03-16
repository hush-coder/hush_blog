# several question

## 1. Permute vs. Transpose

| 层面                     | Permute                                   | Transpose                                      |
|--------------------------|-------------------------------------------|------------------------------------------------|
| Python API (PyTorch)     | 任意维度重排                              | 仅交换两个维度                                 |
| Python API (TensorFlow)  | -                                         | 通过perm参数实现任意重排                        |
| MLIR层面                 | 显式的维度重排操作                        | 转置操作，可能被优化消除                       |
| 数学本质                 | 轴的任意排列                              | 轴的特定交换（二维矩阵的转置是其特例）         |
| 优化潜力                 | 必须执行                                  | 恒等变换时可消除                               |

## 2. 逻辑视图 vs. 实际数据搬运


| 层面               | 内存视角（逻辑视图）                               | 实际内存搬运（数据重排）                                     |
| :----------------- | :------------------------------------------------- | :----------------------------------------------------------- |
| 定义               | 仅改变张量的维度顺序解释，数据位置不变             | 按新顺序重新排列数据在内存中的位置                           |
| 实现方式           | 修改元数据（形状、步长）                           | 内存拷贝、SIMD shuffle、专用硬件指令                         |
| 代价               | 几乎为零（仅CPU计算偏移）                           | 与数据量成正比，可能成为性能瓶颈                             |
| 触发条件           | 默认操作（如`permute`、`transpose`返回视图）        | 需要连续内存时（显式或隐式）                                 |
| 框架示例           | PyTorch中`permute`返回视图，共享存储                | `contiguous()` 触发实际拷贝                                  |
| 硬件视角           | 仍为原内存，但访问模式改变（可能引起缓存未命中）   | 数据被物理移动到新位置，布局符合后续操作                     |

### 2.1 视图（View）——仅改变元数据

在高级框架（如PyTorch、NumPy）中，permute 或 transpose 通常返回原始数据的视图（view），即不复制数据，只修改张量的元数据（如形状、步长）。例如：

```python
import torch
x = torch.randn(2, 3, 4)      # 形状 (2,3,4), 步长 (12,4,1)  (假设行主序)
y = x.permute(2, 0, 1)        # 形状 (4,2,3), 步长 (1,12,4)
```
> **内存视角**：`y` 和 `x` 指向同一块内存，但访问方式变了。`y[i][j][k]` 实际访问的是 `x[k][j][i]`（根据步长计算偏移）。

***优点***

- **零拷贝开销**：不涉及数据移动，操作几乎是瞬时的（`O(1)` 复杂度）。
- **节省内存带宽**：不消耗内存读写带宽，对大型张量尤其重要。
- **避免重复搬运**：同一个视图可以被多个操作共享，减少冗余拷贝。

***缺点***

- **非连续内存访问**：改变视图始终无法改变访问某维度的步长，使得后续操作（如卷积、矩阵乘）无法利用缓存局部性，可能引发大量的缓存未命中，降低性能。
- **隐式拷贝风险**：许多底层算子（如 `view`、`flatten`、某些卷积实现）要求输入张量在内存中连续。若传递非连续视图，框架可能会自动插入 `contiguous()` 操作，触发隐式的内存搬运，此时性能反而更差（因为搬运发生在不经意间）。
- **地址计算开销**：每次元素访问都需要根据步长计算偏移，比连续内存的指针递增多出额外整数运算。
- **硬件不友好**：SIMD 向量化通常要求连续或固定步长的数据，非连续布局可能导致无法使用高效的 SIMD 指令，迫使编译器回退到标量循环。

### 2.2 实际搬运——当需要连续内存时

如果后续操作要求张量在内存中连续（如大多数卷积、矩阵乘法），那么视图可能无法直接使用，必须调用 `.contiguous()` 方法，此时会实际复制数据，按新顺序重新排列内存。

```python
z = y.contiguous()   # 实际搬运数据，内存变为连续
```

***原理***

- 分配新内存，按照新维度顺序将数据从原位置复制到新位置。
- 结果张量通常是内存连续的（步长与形状匹配）。

***优点***

- **连续内存布局**：后续访问具有空间局部性，缓存命中率高，尤其适合大规模计算（如矩阵乘法、卷积）。
- **便于向量化**：连续数据可以自然地映射到 SIMD 指令，提高计算吞吐量。
- **确定性**：开发者明确知道数据布局，避免框架自动插入隐式拷贝带来的不确定性。
- **硬件优化机会**：某些硬件（如 GPU 的 Tensor Cores）要求特定的内存布局（如 NHWC 或 NCHW），显式搬运可以提前适配。

***缺点***

- **显式拷贝开销**：需要读写整个张量的数据，耗时与数据量成正比（O(n) 复杂度），且消耗内存带宽。
- **增加内存占用**：临时分配新内存（除非可以原地转置，但通常很难实现），可能导致峰值内存上升。
- **可能冗余**：如果后续操作很快又将布局改回，则搬运可能白费。

### 2.3 权衡取舍

***(1) 数据规模***

- **小张量**：视图的地址计算开销可能超过拷贝开销，此时直接搬运可能更简单。
- **大张量**：视图通常更优，因为它避免了巨大的 I/O 开销。但若后续大量操作都受益于连续布局，则一次搬运可能值得。

***(2) 后续操作的性质***

- **操作对连续性敏感**：如果后续是一连串的逐元素操作（如加法、激活函数），它们通常不要求连续性，视图即可高效工作。
- **操作需要连续输入**：如果后续是矩阵乘、卷积、view 等，则最好提前搬运一次，避免多次隐式拷贝。
- **操作链长度**：如果重排后的视图只被少量操作使用，视图更佳；如果被大量算子复用，搬运可能更好。

***(3) 硬件架构***

- **CPU**：缓存层级复杂，非连续访问可能导致严重的缓存未命中，因此倾向于在必要时搬运以保持连续性。
- **GPU**：显存带宽高，但非合并访问（coalesced access）对性能至关重要。视图可能导致非合并访问，极大降低吞吐量，因此 GPU 上常常需要显式重排（如通过共享内存转置）来保证合并访问。
- **专用加速器（NPU/TPU）**：可能固定要求某种数据布局，必须提前搬运。

***(4) 框架/编译器的智能优化***

现代框架和编译器（如 PyTorch、TensorFlow、XLA、MLIR）会尝试自动做出权衡：

- **惰性策略**：默认返回视图，仅在必要时插入 contiguous()（例如当用户调用 view 且不连续时）。
- **算子融合**：编译器可能将转置与后续操作融合，例如将 transpose + conv 融合为 conv 的一个变体，避免显式搬运。
- **布局优化**：推理引擎（如 TensorRT、OpenVINO）会在模型编译阶段分析整个图，决定最优的全局数据布局，并插入必要的重排操作。

***(5) 编程意图与可读性***

- 使用视图更符合函数式编程风格，减少副作用。
- 显式搬运让代码更易于理解数据布局，避免隐式行为。

## 3. 转置下沉

***相关代码***

```cpp
// 第396-398行：收集所有使用者的信息
SmallVector<std::pair<Operation *, unsigned>> users;
for (OpOperand &use : result.getUses()) {
    users.push_back({use.getOwner(), use.getOperandNumber()});  // 记录：谁用了，用在第几个操作数
}

// 第400-409行：对每个使用者都复制一份Transpose
int64_t user_id = 0;
for (auto [user, operandIndex] : users) {  // 逐个遍历使用者
    // 创建新的Transpose，放在原始Transpose之后
    rewriter.setInsertionPointAfter(op);  // 在原始op之后插入
    auto newTransposeOp = rewriter.create<top::TransposeOp>(
        new_transpose_Loc,           // 新的位置标记
        op->getResultTypes(),        // 输出类型
        op->getResult(0),            // 输入：还是用op的结果
        rewriter.getI64ArrayAttr(perm)  // perm参数：保持不变
    );
    
    // 将使用者的输入改成新的Transpose输出
    user->setOperand(operandIndex, newTransposeOp->getResult(0));
    user_id++;
}

// 第410行：最后，删除原始的Transpose
rewriter.replaceOp(op, {op.getInput()});  // 用Transpose的输入直接替换Transpose
```

***初始MLIR代码***

```mlir
%0 = "top.Input"() : () -> tensor<10x20xf32>
%1 = "top.Transpose"(%0) {perm = [1, 0]} : (tensor<10x20xf32>) -> tensor<20x10xf32>

%2 = "top.Add"(%1, %cst) : (tensor<20x10xf32>, tensor<20x10xf32>) -> tensor<20x10xf32>
%3 = "top.Mul"(%cst, %1) : (tensor<20x10xf32>, tensor<20x10xf32>) -> tensor<20x10xf32>
%4 = "top.Concat"(%1, %2, %3) {axis = 0} : (tensor<20x10xf32>, tensor<20x10xf32>, tensor<20x10xf32>) -> tensor<60x10xf32>
```

%1 是一个 Transpose，被 3 个不同的操作使用：
1. Add 的第 0 个输入
2. Mul 的第 1 个输入
3. Concat 的第 0 个输入

### 3.1 第 1 步：收集使用者信息（第 396-398 行）

```cpp
SmallVector<std::pair<Operation *, unsigned>> users;
for (OpOperand &use : result.getUses()) {
    users.push_back({use.getOwner(), use.getOperandNumber()});
}
```

执行过程

```bash
遍历%1的所有使用者：

use_1: OpOperand 属于 Add
  - use.getOwner() = Add 操作
  - use.getOperandNumber() = 0  (Add的第0个输入位置)
  - 记录：{Add, 0}

use_2: OpOperand 属于 Mul
  - use.getOwner() = Mul 操作
  - use.getOperandNumber() = 1  (Mul的第1个输入位置)
  - 记录：{Mul, 1}

use_3: OpOperand 属于 Concat
  - use.getOwner() = Concat 操作
  - use.getOperandNumber() = 0  (Concat的第0个输入位置)
  - 记录：{Concat, 0}

最终 users = [
  {Add操作, 0},
  {Mul操作, 1},
  {Concat操作, 0}
]
```

### 3.2 第 2 步：为每个使用者复制 Transpose

```cpp
//********** 第一次迭代（user_id = 0） **********//

auto [user, operandIndex] = users[0];  // {Add, 0}

std::string new_transpose_loc = "Transpose_0";  // 源位置标记+序号
auto new_transpose_Loc = NameLoc::get("Transpose_0");

rewriter.setInsertionPointAfter(op);  // 在原始Transpose后插入
auto newTransposeOp = rewriter.create<top::TransposeOp>(
    new_transpose_Loc,
    op->getResultTypes(),           // tensor<20x10xf32>
    op->getResult(0),               // 输入还是 %1（Transpose的结果）
    rewriter.getI64ArrayAttr({1,0}) // perm = [1,0]
);
// 创建了新的Transpose，记为 %1_copy_0

user->setOperand(operandIndex, newTransposeOp->getResult(0));
// Add 的第 0 个输入：从 %1 改成 %1_copy_0
```

```mlir
// 此时 MLIR 变成：

%1 = "top.Transpose"(%0) {perm = [1, 0]} : ...
%1_copy_0 = "top.Transpose"(%1) {perm = [1, 0]} : ...

%2 = "top.Add"(%1_copy_0, %cst) : ...  ← 已改！
%3 = "top.Mul"(%cst, %1) : ...         ← 还未改
%4 = "top.Concat"(%1, %2, %3) : ...    ← 还未改
```

```cpp
//********** 第二次迭代（user_id = 1） **********//
auto [user, operandIndex] = users[1];  // {Mul, 1}

std::string new_transpose_loc = "Transpose_1";
auto newTransposeOp = rewriter.create<top::TransposeOp>(
    new_transpose_Loc,
    op->getResultTypes(),
    op->getResult(0),               // 输入还是 %1
    rewriter.getI64ArrayAttr({1,0})
);
// 创建了新的Transpose，记为 %1_copy_1

user->setOperand(operandIndex, newTransposeOp->getResult(0));
// Mul 的第 1 个输入：从 %1 改成 %1_copy_1
```

```mlir
// 此时 MLIR 变成：

%1 = "top.Transpose"(%0) {perm = [1, 0]} : ...
%1_copy_0 = "top.Transpose"(%1) {perm = [1, 0]} : ...
%1_copy_1 = "top.Transpose"(%1) {perm = [1, 0]} : ...

%2 = "top.Add"(%1_copy_0, %cst) : ...    ← 已改
%3 = "top.Mul"(%cst, %1_copy_1) : ...    ← 已改！
%4 = "top.Concat"(%1, %2, %3) : ...      ← 还未改
```

```cpp
//********** 第三次迭代（user_id = 2） **********//
auto [user, operandIndex] = users[2];  // {Concat, 0}

std::string new_transpose_loc = "Transpose_2";
auto newTransposeOp = rewriter.create<top::TransposeOp>(
    new_transpose_Loc,
    op->getResultTypes(),
    op->getResult(0),               // 输入还是 %1
    rewriter.getI64ArrayAttr({1,0})
);
// 创建了新的Transpose，记为 %1_copy_2

user->setOperand(operandIndex, newTransposeOp->getResult(0));
// Concat 的第 0 个输入：从 %1 改成 %1_copy_2
```

```mlir
// 此时 MLIR 变成：

%1 = "top.Transpose"(%0) {perm = [1, 0]} : ...
%1_copy_0 = "top.Transpose"(%1) {perm = [1, 0]} : ...
%1_copy_1 = "top.Transpose"(%1) {perm = [1, 0]} : ...
%1_copy_2 = "top.Transpose"(%1) {perm = [1, 0]} : ...

%2 = "top.Add"(%1_copy_0, %cst) : ...       ← 已改
%3 = "top.Mul"(%cst, %1_copy_1) : ...       ← 已改
%4 = "top.Concat"(%1_copy_2, %2, %3) : ...  ← 已改！
```

### 3.3 第 3 步：删除原始 Transpose

```cpp
rewriter.replaceOp(op, {op.getInput()});
//                     ↑ op.getInput() = %0
```

**这句话做什么？**

1. 删除原始的 %1
2. 用 %0（Transpose 的输入）替换所有剩余对 %1 的引用

但注意：此时 `%1` 已经没有使用者了！ 因为我们已经把所有使用者都改成了指向新的 Transpose。

**最终结果**

```mlir
%0 = "top.Input"() : () -> tensor<10x20xf32>

%1_copy_0 = "top.Transpose"(%0) {perm = [1, 0]} : (tensor<10x20xf32>) -> tensor<20x10xf32>
%1_copy_1 = "top.Transpose"(%0) {perm = [1, 0]} : (tensor<10x20xf32>) -> tensor<20x10xf32>
%1_copy_2 = "top.Transpose"(%0) {perm = [1, 0]} : (tensor<10x20xf32>) -> tensor<20x10xf32>

%2 = "top.Add"(%1_copy_0, %cst) : (tensor<20x10xf32>, tensor<20x10xf32>) -> tensor<20x10xf32>
%3 = "top.Mul"(%cst, %1_copy_1) : (tensor<20x10xf32>, tensor<20x10xf32>) -> tensor<20x10xf32>
%4 = "top.Concat"(%1_copy_2, %2, %3) {axis = 0} : (tensor<20x10xf32>, tensor<20x10xf32>, tensor<20x10xf32>) -> tensor<60x10xf32>

// 原始的 %1 已被删除，所有使用者现在用自己的 Transpose 副本
```

***核心逻辑总结***

1. **查找范围**：只查一层直接使用 ✓
2. **数量限制**：useCount 和 tolerable_branches 限制范围
3. **互相消除判断**：user_perm[perm[i]] == i 对所有i成立
4. **沉底操作**：
    - 复制Transpose到每个使用处
    - 删除原始Transpose
    - 让使用者的输入改为新复制的Transpose输出
5. **最终效果**：原始Transpose被"推"到使用处，可能被后续优化消除

# Layout Conversion

# 阶段一：通道维度后移

## 具体操作

```
NCHW -> NHWC
```

| 步骤 | 操作                     | 说明                                                       |
|:----:|--------------------------|------------------------------------------------------------|
| ①    | `__wait(...)`            | 等待输入数据从 (m,n) 块准备就绪                            |
| ②    | `__block2dref_load()`    | 将输入块 `input[m*BS][n*BS]` 加载到硬件缓冲区              |
| ③    | `__transpose()`          | 硬件执行转置：BS×BS → BS×BS（行列互换）                    |
| ④    | `__block2dref_store()`   | 存储转置后的块到输出 `output[n*BS][m*BS]`                  |
| ⑤    | `__notify(...)`          | 通知下一个操作该块已准备好                                 |

## 三大核心原因：

### 原因一：空间局部性 (Spatial Locality)

***实际卷积计算场景***

假设一个 3×3 卷积,输入是 RGB 图像:

```cpp
// 卷积核形状: [3×3×3] (3x3空间, 3通道)
// 计算输出像素 out[0,0] 需要读取输入的 3×3×3 = 27 个数值

```

***NHWC 内存布局 (连续访问)***

```
内存地址:  0    1    2    3    4    5    6    7    8
数据:     [R00][G00][B00][R01][G01][B01][R02][G02][B02]  ← 第0行
          [R10][G10][B10][R11][G11][B11][R12][G12][B12]  ← 第1行
          [R20][G20][B20][R21][G21][B21][R22][G22][B22]  ← 第2行

3×3 卷积读取:
位置(0,0): 地址 0-2   [R00,G00,B00]  ← 连续!
位置(0,1): 地址 3-5   [R01,G01,B01]  ← 连续!
位置(0,2): 地址 6-8   [R02,G02,B02]  ← 连续!
...
```
总共 9 个位置,每个位置读取连续的 3 个通道值

**代码示例:**

```cpp
// NHWC 卷积 - 内存访问连续
float conv_nhwc(float* input, float* kernel) {
    float sum = 0;
    // input: [H][W][C] 布局
    for (int kh = 0; kh < 3; kh++) {
        for (int kw = 0; kw < 3; kw++) {
            // 一次读取连续的 RGB 三个值
            float* pixel = &input[(kh * width + kw) * 3];  // 基地址
            for (int c = 0; c < 3; c++) {
                sum += pixel[c] * kernel[kh][kw][c];  // pixel[0,1,2] 连续!
            }
        }
    }
    return sum;
}
```

***NCHW 内存布局 (跳跃访问)***

```
内存地址:     0    1    2    3    4  ...  50   51   52  ...  100  101  102
R 通道全部:  [R00][R01][R02][R10][R11]... [R20][R21][R22]
G 通道全部:                                              [G00][G01][G02]...
B 通道全部:                                                              [B00]...

3×3 卷积读取:
R 通道: 地址 0, 1, 2, 50, 51, 52, 100, 101, 102  ← 跳跃 50 个位置!
G 通道: 地址 224, 225, 226, ...                  ← 又跳 124 个位置!
B 通道: 地址 448, 449, 450, ...                  ← 又跳 224 个位置!
```

**代码示例:**

```cpp
// NCHW 卷积 - 内存访问跳跃
float conv_nchw(float* input, float* kernel) {
    float sum = 0;
    int channel_stride = height * width;  // 224 * 224 = 50176!
    
    // input: [C][H][W] 布局
    for (int c = 0; c < 3; c++) {
        float* channel = &input[c * channel_stride];  // 跨越大块内存!
        for (int kh = 0; kh < 3; kh++) {
            for (int kw = 0; kw < 3; kw++) {
                sum += channel[kh * width + kw] * kernel[c][kh][kw];
            }
        }
    }
    return sum;
}
```
***内存访问距离对比:***

- **NHWC**: 相邻数据间隔 = 1 个 float (4 bytes)
- **NCHW**: 相邻通道间隔 = 224×224 个 float (200 KB!)

### 原因二：SIMD 向量化效率

***CPU 向量指令的工作原理***

现代 CPU 有 SIMD (Single Instruction Multiple Data) 指令:

- **ARM NEON**: 128-bit 寄存器,一次处理 4 个 float
- **x86 AVX2**: 256-bit 寄存器,一次处理 8 个 float
- **x86 AVX512**: 512-bit 寄存器,一次处理 16 个 float

***NHWC 向量化示例 (ARM NEON)***

```cpp
// NHWC: [R0,G0,B0,A0, R1,G1,B1,A1, R2,G2,B2,A2, R3,G3,B3,A3]
//       |←  128-bit  →|←  128-bit  →|←  128-bit  →|←  128-bit  →|

#include <arm_neon.h>

void process_4_pixels_nhwc(float* input, float* kernel, float* output) {
    // 一次加载 4 个像素的 RGBA 数据 (16 个 float,连续!)
    float32x4_t pixel0 = vld1q_f32(&input[0]);   // [R0,G0,B0,A0]
    float32x4_t pixel1 = vld1q_f32(&input[4]);   // [R1,G1,B1,A1]
    float32x4_t pixel2 = vld1q_f32(&input[8]);   // [R2,G2,B2,A2]
    float32x4_t pixel3 = vld1q_f32(&input[12]);  // [R3,G3,B3,A3]
    
    // 一次加载 4 个卷积核权重
    float32x4_t weight = vld1q_f32(kernel);
    
    // 4 个像素并行计算!
    float32x4_t result0 = vmulq_f32(pixel0, weight);
    float32x4_t result1 = vmulq_f32(pixel1, weight);
    float32x4_t result2 = vmulq_f32(pixel2, weight);
    float32x4_t result3 = vmulq_f32(pixel3, weight);
    
    // 一次性写回结果
    vst1q_f32(&output[0], result0);
    vst1q_f32(&output[4], result1);
    // ... 4 条指令处理 16 个数据!
}
```

**效率:**

- 标量代码: 16 个 mul 指令
- NEON 代码: 4 个 vmulq_f32 指令 → 快 4 倍

***NCHW 向量化问题***

```cpp
// NCHW: R 通道 [R0,R1,R2,R3, ...] 距离 G 通道 50KB!
//       无法同时加载同一像素的多个通道

void process_4_pixels_nchw(float* input, float* kernel, float* output) {
    int stride = 224 * 224;  // 50176
    
    // 加载 4 个 R 通道值 (连续)
    float32x4_t r_vals = vld1q_f32(&input[0]);  // [R0,R1,R2,R3]
    
    // 但是 G 通道在 50KB 之外!
    float32x4_t g_vals = vld1q_f32(&input[stride]);     // cache miss!
    float32x4_t b_vals = vld1q_f32(&input[stride * 2]); // cache miss!
    
    // 需要手动重排数据才能做像素级计算,额外开销!
    // 或者只能按通道处理,失去像素间的并行性
}
```

***实际性能测试***

```cpp
// 测试代码: 处理 224×224×3 图像
void benchmark() {
    // NHWC 布局
    float input_nhwc[224][224][3];  // 连续的 RGB
    auto start = now();
    for (int h = 0; h < 224; h++) {
        for (int w = 0; w < 224; w += 4) {  // 每次处理 4 个像素
            float32x4x3_t pixels = vld3q_f32(&input_nhwc[h][w][0]);
            // vld3q_f32: 专门的"交错加载"指令,完美匹配 NHWC!
            // 自动将 [R,G,B,R,G,B,...] 分离成 3 个向量
        }
    }
    print("NHWC: ", elapsed(start));  // 输出: 12 ms
    
    // NCHW 布局
    float input_nchw[3][224][224];  // 分离的通道
    start = now();
    for (int h = 0; h < 224; h++) {
        for (int w = 0; w < 224; w += 4) {
            float32x4_t r = vld1q_f32(&input_nchw[0][h][w]);  // 加载 R
            float32x4_t g = vld1q_f32(&input_nchw[1][h][w]);  // 不连续!
            float32x4_t b = vld1q_f32(&input_nchw[2][h][w]);
        }
    }
    print("NCHW: ", elapsed(start));  // 输出: 35 ms
}
```
> 这里在后面还需要深入到底层

### 原因三：缓存命中率

***CPU 缓存层次结构***

```
L1 Cache: 32 KB,  延迟 ~4 cycles    ← 最快
L2 Cache: 256 KB, 延迟 ~12 cycles
L3 Cache: 8 MB,   延迟 ~40 cycles
RAM:      延迟 ~200 cycles          ← 最慢
```

**Cache Line**: CPU 每次从内存加载 64 字节 (16 个 float)

***NHWC 缓存友好示例***

```cpp
// 3×3 卷积,处理 224×224×3 图像

// NHWC 内存布局
float input_nhwc[224][224][3];  // [H][W][C]

// 访问 3×3 窗口
for (int kh = 0; kh < 3; kh++) {
    for (int kw = 0; kw < 3; kw++) {
        // 访问 input[10+kh][10+kw][0:3]
        float* ptr = &input_nhwc[10+kh][10+kw][0];
        float r = ptr[0];  // 地址 X
        float g = ptr[1];  // 地址 X+4    ← 同一 cache line!
        float b = ptr[2];  // 地址 X+8    ← 同一 cache line!
    }
}

// 内存访问模式:
// 第一次读取触发 cache line 加载 (64 字节 = 5 个像素):
// [R10,10][G10,10][B10,10][R10,11][G10,11][B10,11][R10,12][G10,12][B10,12]...
//  ↑ 读这个        ↑ 已在缓存     ↑ 已在缓存     ↑ 下次读取也命中!

// Cache miss 次数: 
// 9 个位置,每行 ~2 次 miss,共约 6 次 miss
```

***NCHW 缓存不友好示例***

```cpp
// NCHW 内存布局
float input_nchw[3][224][224];  // [C][H][W]

// 访问 3×3 窗口的 RGB
for (int c = 0; c < 3; c++) {
    for (int kh = 0; kh < 3; kh++) {
        for (int kw = 0; kw < 3; kw++) {
            float val = input_nchw[c][10+kh][10+kw];
        }
    }
}

// 内存访问模式:
// R 通道: 地址 0x0000 ~ 0x0023      ← cache line 加载
// G 通道: 地址 0xC000 ~ 0xC023      ← 距离 R 通道 50KB,完全不同的 cache line!
// B 通道: 地址 0x18000 ~ 0x18023    ← 又是新的 cache line!

// Cache miss 次数:
// 每个通道 9 个位置,每个通道约 6 次 miss
// 3 个通道共约 18 次 miss!  ← 比 NHWC 多 3 倍!
```

# 阶段二：Sink Transpose

## 包含Pattern

```
populateConvertTransposePatterns
  ├─ ConvertTransposeToReshapeOp
  ├─ RemoveRedundantShapeOp<ReshapeOp, ReshapeOp>
  └─ populateFoldConstantMemoryPatterns
      ├─ FoldConstantTransposeOp
      └─ FoldConstantReshapeOp

populateEraseRedundantTransposeReshapePatterns
  ├─ RemoveRedundantTransposeOp
  └─ RemoveRedundantShapeOp<ReshapeOp, ReshapeOp>
```

## 优化流程

```
以下流程迭代八次

SinkTranspose Pattern 的真正执行流程：

Step 1: 数一下有多少个使用者
        useCount = ...

Step 2: 判断是否绝大多数都是"可消除的"（第一段）
        if (useCount > 1) {
            检查：有多少个使用者本身是Transpose且与当前Transpose互相消除？
            if (eliminable_users_num >= useCount - 1) {
                → 下沉Transpose（复制到每个使用者）
                → 返回success，进入下一迭代
            }
        }

Step 3: 否则，判断是否所有使用者都是layout不敏感的（第二段）
        if (useCount > 1 && useCount <= 2) {
            检查：有多少个使用者是layout agnostic/transform/mixed？
            if (sinkable_users_num == useCount) {
                → 分裂Transpose（复制到每个使用者）
                → 返回success，进入下一迭代
            }
        }

Step 4: 否则，如果只有1个使用者，进行分类处理（第三段）
        if (result.hasOneUse()) {
            switch(使用者的分类) {
                case LAYOUT_SENSITIVE: ...
                case LAYOUT_AGNOSTIC: ...
                case LAYOUT_TRANSFORM: ...
                case LAYOUT_MIXED: ...
            }
        }
```

```
输入MLIR IR
    ↓
SinkTranspose (分类处理不同使用场景)
    ├─ 如果是多个可互相消除的Transpose → 复制+配对消除
    ├─ 如果是多个Layout Agnostic使用 → 复制+后移
    └─ 如果是单一使用
        ├─ LAYOUT_SENSITIVE: 直接优化（通常不处理）
        ├─ LAYOUT_AGNOSTIC: optimizeLayoutAgnosticOp
        ├─ LAYOUT_TRANSFORM: optimizeLayoutTransformOp (Reshape/Pad/Slice/Expand)
        └─ LAYOUT_MIXED: optimizeLayoutMixedOp (有轴参数)
    ↓
MoveTransposeAfterConcat (优化Concat + Transpose)
    ↓
MergeEquivalentTransposeOp (消除重复Transpose)
    ↓
输出优化后的IR
```

## 优化策略1：消除Transpose

消除代码中互相抵消的Transpose操作

### 什么是互相抵消的Transpose？

```mlir
%1 = top.Transpose(%input) {perm=[0,2,3,1]}   // 变换1：NCHW → NHWC
%2 = top.Transpose(%1) {perm=[0,3,1,2]}       // 变换2：NHWC → NCHW（恢复！）

// 这两个操作抵消了，应该直接删除
```

***数学角度：***

- 如果 perm1 和 perm2 满足：perm1 ∘ perm2 = 恒等映射（即组合后等于[0,1,2,3]）
- 那这两个Transpose就可以消除

***判断方法（代码中）：***

```cpp
// 检查：perm[perm_inverse[i]] == i 对所有i都成立
bool canMutuallyEliminate = (user_perm[intermediate_dim] == i);
```

### 优化前的场景

```mlir
%1 = top.Transpose(%input) {perm=[0,2,3,1]}   // NCHW → NHWC

// 后续有多个地方用%1的结果
%2 = top.Transpose(%1) {perm=[0,3,1,2]}  // 使用方1：再转回NCHW（互相抵消！）
%3 = top.Add(%1, %bias)                  // 使用方2：直接用NHWC数据
%4 = top.Transpose(%1) {perm=[0,3,1,2]}  // 使用方3：也想转回NCHW（互相抵消！）
```

***问题：***

- 有多个Transpose操作，其中某些互相抵消
- 这些都是昂贵的操作（涉及大量内存重组）
- 我们应该消除掉这些无用的操作

### 策略条件

***第一步：检查是否值得优化***

```cpp
size_t useCount = std::distance(result.use_begin(), result.use_end());
// useCount = 3（%2、%3、%4都用了%1）

if (useCount > 1) {  // 只有多个使用才值得优化
    // 继续...
}
```

***第二步：统计有多少个使用可以被消除***

```cpp
int64_t eliminable_users_num = 0;

for (OpOperand &use : result.getUses()) {
    Operation *user = use.getOwner();  // 谁在用这个%1？
    
    if (isa<top::TransposeOp>(user)) {  // 使用者本身是Transpose吗？
        // 检查这个Transpose是否与原来的互相抵消
        bool Eliminable = (user_perm[intermediate_dim] == i);
        if (Eliminable) eliminable_users_num++;
    }
}
```
在我们的例子中：

1. `%2`使用者是`top.Transpose`，能互相抵消 ✅ `eliminable_users_num++`
2. `%3`使用者是`top.Add`，不是`Transpose` ❌
3. `%4`使用者是`top.Transpose`，能互相抵消 ✅ `eliminable_users_num++`
4. 结果：`eliminable_users_num = 2`

***第三步：判断是否进行优化***

```
if (eliminable_users_num >= (useCount - 1)) {
    // 进行优化
}
```
**翻译**：如果至少有 (useCount-1) 个使用可以被消除，就优化

在我们的例子中：

- useCount = 3，需要至少 3-1=2 个可被消除
- eliminable_users_num = 2，正好满足！✅
- 进行优化

## 优化策略2：分裂Transpse给Element-wise操作

### 优化条件

```cpp
if (useCount > 1 && useCount <= 2) {
    if (sinkable_users_num == useCount) {  // 所有都是layout agnostic
```

### 优化过程

## 优化策略3：单独组合分case处理

### Case 1: LAYOUT_SENSITIVE（不处理）

**定义**：对数据排列敏感的操作（Conv、MatMul等）

**为什么不处理？**因为这些操作依赖于特定的数据排列格式

**代码：**

```cpp
case LayoutCategory::LAYOUT_SENSITIVE:
    return optimizeLayoutSensitiveOp(user, rewriter);  // 直接返回failure()
```

### Case 2: LAYOUT_AGNOSTIC（下沉）

**定义**：对数据排列不敏感的操作（Add、Mul、ReLU等）

**操作**：optimizeLayoutAgnosticOp 函数（第28-124行）

**逻辑**：

```cpp
// 1. 检查：使用者(比如Add)的输入是来自Transpose吗？
if (isa<TransposeOp>(opOperandDefiningOp)) {
    transposeOp = llvm::dyn_cast<TransposeOp>(opOperandDefiningOp);
}

// 2. 如果是，进行"沉底"
// 创建新Transpose放在操作之后
auto newTransposeOp = rewriter.create<top::TransposeOp>(
    newTransposeLoc, 
    outputType, 
    op->getResult(0),              // 操作的输出
    permAttr                       // 保持原来的perm
);

// 3. 改变操作的输入为原始input
op->setOperand(i, preOp->getOperand(0));  // 使用Transpose的输入
```

效果：

```mlir
// 优化前
%1 = top.Transpose(%input) {perm=[0,2,3,1]}
%2 = top.Add(%1, %bias)

// 优化后
%2_no_perm = top.Add(%input, %bias)
%2 = top.Transpose(%2_no_perm) {perm=[0,2,3,1]}
```

### Case 3: LAYOUT_TRANSFORM（融合）

**定义**：会改变张量形状的操作（Reshape、Pad、Slice、Expand）

**操作**：optimizeLayoutTransformOp 函数（第126-307行）

分4个子操作：

***A. Reshape + Transpose融合（127-227行）***

```mlir
// 优化前
%1 = top.Transpose(%input) {perm=[0,2,3,1]}
%2 = top.Reshape(%1) {shape=[...]}

// 优化后（如果可以融合）
%2_new = top.Reshape(%input) {shape=[...modified...]}
%2 = top.Transpose(%2_new) {perm=[...new_perm...]}
```

***B. Pad + Transpose融合（229-259行）***

```mlir
// 优化前
%1 = top.Transpose(%input) {perm=[0,2,3,1]}
%2 = top.Pad(%1, paddings=[...])

// 优化后
%paddings_adjusted = adjustByPerm(paddings, inverse_perm)
%2_new = top.Pad(%input, paddings=%paddings_adjusted)
%2 = top.Transpose(%2_new) {perm=[0,2,3,1]}
```

***C. Slice + Transpose融合（261-280行）***

```mlir
// 优化前
%1 = top.Transpose(%input) {perm=[0,2,3,1]}
%2 = top.Slice(%1, axes=[2], starts=[5], ends=[10])

// 优化后
%new_axes = applyPermToAxes(axes, perm)
%2_new = top.Slice(%input, axes=%new_axes, ...)
%2 = top.Transpose(%2_new) {perm=[0,2,3,1]}
```

***D. Expand + Transpose融合（282-303行）***

```mlir
// 优化前
%1 = top.Transpose(%input) {perm=[0,2,3,1]}
%2 = top.Expand(%1, size=[1, 224, 224, 3])

// 优化后
%new_size = reorder(size, inverse_perm)
%2_new = top.Expand(%input, size=%new_size)
%2 = top.Transpose(%2_new) {perm=[0,2,3,1]}
```

### Case 4: LAYOUT_MIXED（轴调整）

**定义**：有轴参数（axis/axes）的操作（ReduceMean、ReduceSum等）

**操作**：optimizeLayoutMixedOp 函数（第309-363行）

***逻辑：***

```cpp
// 1. 获取操作的轴参数
auto axis = getAxesIdx(op, inputShape.size())[0];

// 2. 通过perm变换轴坐标
int64_t newAxis = perm[axis];

// 3. 更新操作的轴参数
updateOpAxes(op, newAxis);

// 4. 删除原始Transpose，添加新Transpose
```

***效果：***

```mlir
// 优化前
%1 = top.Transpose(%input) {perm=[0,2,3,1]}
%2 = top.ReduceMean(%1, axis=2)

// 优化后
newAxis = perm[2] = 3
%2_new = top.ReduceMean(%input, axis=3)
%2 = top.Transpose(%2_new) {perm=[0,2,3,1]}
```

## 优化策略4：MoveTransposeAfterConcat (优化Concat + Transpose)

### 前置条件

所有Concat的输入都是相同perm的Transpose

### 优化前

```mlir
// 优化前
%input1: [1,3,224,224] NCHW
%input2: [1,3,224,224] NCHW

%t1 = top.Transpose(%input1) {perm=[0,2,3,1]}  // [1,224,224,3] NHWC
%t2 = top.Transpose(%input2) {perm=[0,2,3,1]}  // [1,224,224,3] NHWC

%concat = top.Concat(%t1, %t2, axis=2)  // NHWC格式，沿W维（轴2）拼接
        // 输出：[1, 224, 448, 3]
```

### 优化过程

```cpp
int64_t newConcatAxis = mlir::cast<mlir::IntegerAttr>(permVec[concatAxis]).getInt();
```

***perm = [0, 2, 3, 1] 表示：***

1. 输出维度0来自输入维度0
2. 输出维度1来自输入维度2
3. 输出维度2来自输入维度3
4. 输出维度3来自输入维度1

***总结Concat融合：***

1. 前置：所有输入都是同样Transpose
2. 轴变换：`newAxis = perm[oldAxis]`
3. 先拼接再Transpose

### 优化后

```mlir
// 优化后
%concat_no_t = top.Concat(%input1, %input2, axis=3)  // NCHW格式，沿W维拼接
             // [1,3,224,224] + [1,3,224,224] → [1,3,224,448]

%concat = top.Transpose(%concat_no_t) {perm=[0,2,3,1]}
        // [1,3,224,448] → [1,224,448,3] NHWC格式 ✓
```

## 优化策略5：MergeEquivalentTransposeOp (消除重复Transpose)

```mlir
// 多个Transpose使用同一个输入且perm相同
%t1 = top.Transpose(%x) {perm=[0,2,3,1]}
... 使用%t1 ...
%t2 = top.Transpose(%x) {perm=[0,2,3,1]}
... 使用%t2 ...
  ↓
%shared = top.Transpose(%x) {perm=[0,2,3,1]}
// %t1和%t2都替换为%shared
```

# 阶段三：ConvertTranspose

## 包含Pattern

```
populateConvertTransposePatterns
  ├─ ConvertTransposeToReshapeOp
  ├─ RemoveRedundantShapeOp<ReshapeOp, ReshapeOp>
  └─ populateFoldConstantMemoryPatterns
      ├─ FoldConstantTransposeOp
      └─ FoldConstantReshapeOp

populateEraseRedundantTransposeReshapePatterns
  ├─ RemoveRedundantTransposeOp
  └─ RemoveRedundantShapeOp<ReshapeOp, ReshapeOp>
```

## 整体流程

```mlir
// ===== 经过OptimizeLayoutPass后的IR =====
%const_weight = top.Constant() {value=[...]}  // [64, 3, 3, 3]
%input_t = top.Transpose(%input) {perm=[0,2,3,1]}
%conv_out = top.Conv(%input_t, %const_weight)
%add_out = top.Add(%conv_out, %bias)
%add_t = top.Transpose(%add_out) {perm=[0,3,1,2]}

// ===== ConvertTransposePass 执行 =====

// 第一轮：处理%const_weight之后的Transpose
模式匹配：FoldConstantTransposeOp（虽然常数本身没有Transpose...）

// 第二轮：处理%input_t
模式匹配：
- FoldConstantTransposeOp → %input的Constant吗? ✗
- RemoveRedundantTransposeOp → %input后面是Transpose吗? ✗
- ConvertTransposeToReshapeOp → 满足条件? 
  - perm=[0,2,3,1]，input=[1,3,224,224]
  - nonUnitPermIndices=[2,3,1]，不递增 ✗
- 无法转换，保留

// 第三轮：处理%add_t  
模式匹配：ConvertTransposeToReshapeOp
- input_shape = [1,112,112,64]（NHWC格式，Conv的输出）
- perm = [0,3,1,2]
- nonUnitPermIndices = [3,1,2]（对应output_shape的[H,W,C]）
- 检查递增：3>1 ✗
- 无法转换，保留

// 如果之前有两个连续Transpose
%t1 = top.Transpose(%x) {perm=[0,2,3,1]}
%t2 = top.Transpose(%t1) {perm=[0,3,1,2]}

模式匹配：RemoveRedundantTransposeOp（匹配%t2）
- perm_out = [0,1,2,3]
- 是恒等变换
- 删除%t2，直接用%x

// ===== 最终结果 =====
%const_weight = top.Constant()
%input_t = top.Transpose(%input) {perm=[0,2,3,1]}
%conv_out = top.Conv(%input_t, %const_weight)
%add_out = top.Add(%conv_out, %bias)
%add_t = top.Transpose(%add_out) {perm=[0,3,1,2]}
// （可能还有其他Reshape/Transpose，取决于是否满足条件）
```

## Pattern 1: Transpose -> Reshape

### 条件判断逻辑（第1940-1956行）：

```cpp
// 提取perm中，对应"大小>1维度"的perm值
SmallVector<int64_t> nonUnitPermIndices;
for (auto i : permAttr) {  // i是perm中的每个值
    int64_t permValue = i.getInt();
    if (trans_input_shape[permValue] > 1) {  // 这个维度大小>1
        nonUnitPermIndices.push_back(permValue);  // 记录这个perm值
    }
}

// 检查这些perm值是否递增
for (size_t i = 1; i < nonUnitPermIndices.size(); ++i) {
    if (nonUnitPermIndices[i] <= nonUnitPermIndices[i - 1]) {
        return failure();  // 不递增，无法转换
    }
}
```

### 具体例子：

```mlir
// 例子1：可以转成Reshape ✓
%input: [1, 3, 224, 224]
%t = top.Transpose(%input) {perm=[0,2,3,1]}
```

分析：
- i=0: perm[0]=0, input_shape[0]=1 → 不记录（大小=1）
- i=1: perm[1]=2, input_shape[2]=224 → 记录 perm[1]=2 ✓
- i=2: perm[2]=3, input_shape[3]=224 → 记录 perm[2]=3 ✓
- i=3: perm[3]=1, input_shape[1]=3 → 记录 perm[3]=1 ✓
- nonUnitPermIndices = [2, 3, 1]
- 检查递增：2<3 ✓, 3<1 ✗ → 失败，无法转换

```mlir
// 例子2：可以转成Reshape ✓
%input: [1, 64, 1, 1]
%t = top.Transpose(%input) {perm=[0,3,1,2]}
```
分析：
- i=0: perm[0]=0, input_shape[0]=1 → 不记录
- i=1: perm[1]=3, input_shape[3]=1 → 不记录
- i=2: perm[2]=1, input_shape[1]=64 → 记录 perm[2]=1 ✓
- i=3: perm[3]=2, input_shape[2]=1 → 不记录
- nonUnitPermIndices = [1]
- 只有一个元素，无需检查递增 ✓ 可以转换

```mlir
// 例子3：可以转成Reshape ✓
%input: [1, 1, 224, 224]
%t = top.Transpose(%input) {perm=[0,3,1,2]}
```
分析：
- 大小>1的维度：H(2)=224, W(3)=224
- nonUnitPermIndices = [perm[2], perm[3]] = [1, 2]
- 检查递增：1<2 ✓ → 可以转换！

### 转换过程（第1958-1963行）：

```cpp
// 创建新的Reshape操作，输出形状与Transpose相同
auto new_reshape_op = rewriter.create<ReshapeOp>(
    op->getLoc(), 
    op->getResultTypes(),  // 使用Transpose的输出类型
    op->getOperand(0),     // 使用Transpose的输入
    attributes
);
new_reshape_op.setShapeAttr(rewriter.getI64ArrayAttr(to_shape));  // 设置Reshape的形状
rewriter.replaceOp(op, new_reshape_op.getOutput());  // 替换Transpose
```

为什么这样设计：

- Transpose改变维度视图
- 如果只改变了大小=1的维度的顺序，实际上不需要移动数据
- 可以用更轻量级的Reshape替换

## Pattern 2: FoldConstantTransposeOp

条件（第2179-2181行）：Transpose的输入是Constant

逻辑（第2190-2204行）：

```cpp
// 检查perm是否是恒等变换 [1, 2, 3, ...]
std::vector<int64_t> bypass_perm(dim);
std::iota(bypass_perm.begin(), bypass_perm.end(), 1);  // 生成 [1, 2, 3, ...]

if (*perm != bypass_perm) {
    // perm不是恒等变换，需要重排常数值
    ConstantOp new_const_op;
    if (module::isInteger(const_type)) {
        new_const_op = module::ConvertConstantLayout<APInt>(PreOp, *perm, op->getLoc());
    } else {
        new_const_op = module::ConvertConstantLayout<APFloat>(PreOp, *perm, op->getLoc());
    }
    rewriter.replaceOp(op, new_const_op);  // 用新的Constant替换
    return success();
} else {
    // perm是恒等变换，直接删除Transpose
    rewriter.replaceOp(op, const_output);
    return success();
}
```

### 具体例子：

```mlir
// 例子1：perm不是恒等
%const = top.Constant() {value=[1,2,3,4,5,6,7,8,9,10,11,12]}  // 形状[1,3,2,2]
%t = top.Transpose(%const) {perm=[0,2,3,1]}
```

过程：
- perm = [0,2,3,1]
- bypass_perm = [0,1,2,3]
- perm != bypass_perm ✓
- 调用 ConvertConstantLayout<APFloat>
- 生成新的Constant，值重排为NHWC格式
- 替换Transpose：原Constant删除，新Constant保留，Transpose删除

```mlir
// 例子2：perm是恒等
%const = top.Constant() {value=...}
%t = top.Transpose(%const) {perm=[0,1,2,3]}
```

过程：
- perm = [0,1,2,3]
- bypass_perm = [0,1,2,3]
- perm == bypass_perm ✓
- 直接用const的输出替换Transpose
- Transpose删除，Constant保留，但不再使用Transpose

### 收益：

- 把Transpose的代价从运行时转到编译时（常数折叠）
- 减少运行时Transpose操作

## Pattern 3: RemoveRedundantTransposeOp

条件（第2242-2244行）：当前Transpose的输入是另一个Transpose

逻辑（第2248-2262行）：

```cpp
// 第2250-2251行：组合两个perm
std::vector<int64_t> perm_out;
for (int64_t i = 0; i < dim; i++) {
    perm_out.push_back(pre_perm->at(perm->at(i)));
    // 相当于：perm_out[i] = pre_perm[perm[i]]
}

// 第2253-2254行：生成恒等perm
std::vector<int64_t> bypass_perm(dim);
std::iota(bypass_perm.begin(), bypass_perm.end(), 1);  // [1,2,3,...]

// 第2257-2262行：判断是否能消除
if (perm_out != bypass_perm) {
    // 合并成一个Transpose
    op->setAttr("perm", rewriter.getI64ArrayAttr(perm_out));
    op->setOperand(0, input);  // 输入指向pre_perm的输入
} else {
    // 两个Transpose互相抵消，直接删除
    rewriter.replaceOp(op, input);
}
```

### 具体例子

```mlir
// 例子1：两个Transpose互相抵消
%t1 = top.Transpose(%x) {perm=[0,2,3,1]}      // NCHW→NHWC
%t2 = top.Transpose(%t1) {perm=[0,3,1,2]}     // NHWC→NCHW
```

过程：
- perm = [0,3,1,2]
- pre_perm = [0,2,3,1]
- 计算组合：
  - perm_out[0] = pre_perm[perm[0]] = pre_perm[0] = 0
  - perm_out[1] = pre_perm[perm[1]] = pre_perm[3] = 1
  - perm_out[2] = pre_perm[perm[2]] = pre_perm[1] = 2
  - perm_out[3] = pre_perm[perm[3]] = pre_perm[2] = 3
  - perm_out = [0,1,2,3] 恒等
- 直接删除两个Transpose
- 结果：%t2 = %x

```mlir
// 例子2：两个Transpose需要合并
%t1 = top.Transpose(%x) {perm=[0,2,1,3]}
%t2 = top.Transpose(%t1) {perm=[0,3,1,2]}
```

过程：
- perm = [0,3,1,2]
- pre_perm = [0,2,1,3]
- 计算组合：
  - perm_out[0] = pre_perm[0] = 0
  - perm_out[1] = pre_perm[3] = 3
  - perm_out[2] = pre_perm[1] = 2
  - perm_out[3] = pre_perm[2] = 1
  - perm_out = [0,3,2,1]
- perm_out != [0,1,2,3]
- 改变第二个Transpose的perm为[0,3,2,1]
- 改变第二个Transpose的输入为%x
- 结果：%t2 = top.Transpose(%x) {perm=[0,3,2,1]}
- 删除%t1

### 数学原理

```
组合两个perm的公式：perm_out[i] = pre_perm[perm[i]]

这相当于矩阵乘法在排列上的应用：
pre_perm ∘ perm = 组合后的排列
```

## Pattern 4: RemoveRedundantShapeOp

功能：删除冗余的Reshape操作

```mlir
// 例子
%r1 = top.Reshape(%x) {shape=[1, 224*224, 3]}
%r2 = top.Reshape(%r1) {shape=[1, 224*224, 3]}  // 形状相同
→ 删除%r2，直接用%r1
```

## 总结

1. FoldConstantTransposeOp：把Transpose代价从运行时转到编译时
2. RemoveRedundantTransposeOp：消除冗余的Transpose对（可能在OptimizeLayoutPass中产生）
3. ConvertTransposeToReshapeOp：用轻量级操作替换Transpose（虽然条件苛刻，但能覆盖某些特殊情况）
4. RemoveRedundantShapeOp：清理可能产生的冗余操作