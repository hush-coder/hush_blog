***ONNXRuntime-BatchNormalization篇(1)***

# 摘要

# 目录

# 共享内存优化 ✅

## 前置知识

### 什么是内存访问模式

内存访问模式（Memory Access Pattern）是指程序在访问内存时的空间和时间特征，包括：
- **空间局部性**：访问的内存地址是否连续或邻近
- **时间局部性**：相同内存地址是否被重复访问
- **访问顺序**：内存访问的先后顺序和规律性
- **合并程度**：多个线程是否能够合并访问连续内存

### GPU内存层次结构

```
CPU内存 (最慢，容量最大)
    ↓
GPU全局内存 (慢，容量大)
    ↓
L2缓存 (中等速度，中等容量)
    ↓
L1缓存 (快，小容量)
    ↓
共享内存 (很快，很小容量)
    ↓
寄存器 (最快，最小容量)
```

### BN的内存访问模式

#### 1. BatchNormalization的数据结构
BatchNormalization处理的是4维张量，形状为 (N, C, H, W)：
- N: 批次大小 (Batch Size)
- C: 通道数 (Channels)
- H: 高度 (Height)
- W: 宽度 (Width)

#### 2. 4个参数的含义

BatchNormalization需要4个参数，每个参数都是长度为C的一维数组：
- mean[c]: 通道c的均值
- var[c]: 通道c的方差
- gamma[c]: 通道c的缩放参数
- beta[c]: 通道c的偏移参数

#### 3. 关键理解：通道级别的归一化
BatchNormalization的核心思想是：对每个通道独立进行归一化。

```cpp
// BatchNormalization公式
Y[n,c,h,w] = gamma[c] * ((X[n,c,h,w] - mean[c]) / sqrt(var[c] + epsilon)) + beta[c]
```

**关键点**：对于通道c，无论h和w如何变化，都使用相同的4个参数：
`mean[c], var[c], gamma[c], beta[c]`

#### 4. 具体例子说明
假设我们有一个形状为 (2, 3, 4, 4) 的张量：
- N=2 (2个样本)
- C=3 (3个通道)
- H=4, W=4 (4x4的特征图)

**参数数组**：

```cpp
float mean[3] = {0.1, 0.2, 0.3};    // 3个通道的均值
float var[3] = {0.5, 0.6, 0.7};     // 3个通道的方差
float gamma[3] = {1.0, 1.1, 1.2};   // 3个通道的缩放参数
float beta[3] = {0.0, 0.1, 0.2};    // 3个通道的偏移参数
```

**数据访问模式**：

对于通道0 (c=0)：

```cpp
// 所有 (n, 0, h, w) 位置的元素都使用相同的参数
Y[0,0,0,0] = gamma[0] * ((X[0,0,0,0] - mean[0]) / sqrt(var[0] + eps)) + beta[0];
Y[0,0,0,1] = gamma[0] * ((X[0,0,0,1] - mean[0]) / sqrt(var[0] + eps)) + beta[0];
Y[0,0,0,2] = gamma[0] * ((X[0,0,0,2] - mean[0]) / sqrt(var[0] + eps)) + beta[0];
// ... 所有16个位置都使用 mean[0], var[0], gamma[0], beta[0]

Y[0,0,1,0] = gamma[0] * ((X[0,0,1,0] - mean[0]) / sqrt(var[0] + eps)) + beta[0];
Y[0,0,1,1] = gamma[0] * ((X[0,0,1,1] - mean[0]) / sqrt(var[0] + eps)) + beta[0];
// ... 继续使用相同的4个参数
```

对于通道1 (c=1)：

```cpp
// 所有 (n, 0, h, w) 位置的元素都使用相同的参数
Y[0,0,0,0] = gamma[0] * ((X[0,0,0,0] - mean[0]) / sqrt(var[0] + eps)) + beta[0];
Y[0,0,0,1] = gamma[0] * ((X[0,0,0,1] - mean[0]) / sqrt(var[0] + eps)) + beta[0];
Y[0,0,0,2] = gamma[0] * ((X[0,0,0,2] - mean[0]) / sqrt(var[0] + eps)) + beta[0];
// ... 所有16个位置都使用 mean[0], var[0], gamma[0], beta[0]

Y[0,0,1,0] = gamma[0] * ((X[0,0,1,0] - mean[0]) / sqrt(var[0] + eps)) + beta[0];
Y[0,0,1,1] = gamma[0] * ((X[0,0,1,1] - mean[0]) / sqrt(var[0] + eps)) + beta[0];
// ... 继续使用相同的4个参数
```

#### 5. 代码中的体现
在代码中，我们可以看到这种访问模式：

```cpp
// 计算当前元素属于哪个通道
int c = (idx % (C * H * W)) / (H * W);

// 使用通道c的4个参数
float m = mean[c];      // 通道c的均值
float v = var[c];       // 通道c的方差  
float g = gamma[c];     // 通道c的缩放参数
float b = beta[c];      // 通道c的偏移参数

// 所有属于通道c的元素都使用这4个参数
Y[offset] = g * ((x - m) * inv_std) + b;
```

#### 6. 重复访问的问题
**问题**：同一个通道内的所有元素（N×H×W个）都会重复访问相同的4个参数。

**具体数量**：

- 通道0：2×4×4 = 32个元素，每个都访问 mean[0], var[0], gamma[0], beta[0]
- 通道1：2×4×4 = 32个元素，每个都访问 mean[1], var[1], gamma[1], beta[1]
- 通道2：2×4×4 = 32个元素，每个都访问 mean[2], var[2], gamma[2], beta[2]

**总访问次数**：

- 参数访问：32×4 = 128次（每个通道）
- 全局访问：128×3 = 384次（所有通道）


**完成时间**: 2024年当前时间
**优化内容**:
- 实现了 `_BatchNormalizationSharedMemory` 内核函数
- 添加了 `rocm_batch_norm_shared_memory` C接口函数
- 使用共享内存存储参数，减少全局内存访问

## **具体优化**

### 共享内存策略

1. **共享内存存储**: 使用 `__shared__` 存储 mean, var, gamma, beta 参数
2. **协作加载**: 多个线程协作将参数加载到共享内存
3. **减少全局内存访问**: 计算时从共享内存读取参数，而不是全局内存
4. **同步优化**: 使用 `__syncthreads()` 确保数据加载完成
5. **内存访问优化**: 减少全局内存访问延迟，提升性能

**技术细节**:
```cpp
// 共享内存声明
__shared__ float shared_mean[256];
__shared__ float shared_var[256];
__shared__ float shared_gamma[256];
__shared__ float shared_beta[256];

// 协作加载参数
if (threadIdx.x < C) {
    shared_mean[threadIdx.x] = mean[threadIdx.x];
    shared_var[threadIdx.x] = var[threadIdx.x];
    shared_gamma[threadIdx.x] = gamma[threadIdx.x];
    shared_beta[threadIdx.x] = beta[threadIdx.x];
}
__syncthreads();

// 使用共享内存进行计算
float m = shared_mean[c];
float v = shared_var[c];
float inv_std = rsqrtf(v + epsilon);
Y[offset] = shared_gamma[c] * ((x - m) * inv_std) + shared_beta[c];
```

### 智能选择策略

**重构详情**:
1. **合并内核函数**: 将 `_BatchNormalizationOptimized` 和 `_BatchNormalizationSharedMemory` 合并为 `_BatchNormalizationAdaptive`
2. **自适应逻辑**: 在运行时根据通道数C自动选择实现策略
3. **简化接口**: 将两个C接口函数合并为一个 `rocm_batch_norm`
4. **智能选择**: C >= 256时使用共享内存，C < 256时使用全局内存

**技术实现**:
```cpp
// 自适应策略：根据通道数选择最优实现
if (C >= 256) {
    // 大通道数：使用共享内存优化
    __shared__ float shared_mean[256];
    // ... 共享内存实现
} else {
    // 小通道数：直接使用全局内存
    float m = mean[c];
    // ... 全局内存实现
}
```


## **性能提升分析**:

### **优化前**：
- 全局内存访问：N×C×H×W×4 = 96次（每个元素4次参数访问）
- 延迟：每次访问200-400个时钟周期
### **优化后**：
- 全局内存访问：C×4 = 12次（每个通道4次参数访问）
- 共享内存访问：N×C×H×W×4 = 96次（每个元素4次参数访问）
- 延迟：共享内存访问1-2个时钟周期
### **性能提升**：
- 减少全局内存访问：从96次减少到12次
- 延迟降低：从200-400个时钟周期降低到1-2个时钟周期
- 预期提升：20-30%性能提升
