***ONNXRuntime-basic篇（3）***

# 摘要

# 目录

# 线程与内存的GPU底层

GPU分为若干个SM

GPU总的线程存储在网格当中，网格又被分为若干线程块（blocks），线程块被分为若干线程束（warp），线程束内又被分为若干线程（threads）

## SM

GPU的SM数量是一定的，是并行的基础

SM内的共享内存总量和寄存器总量是一定的

## 线程块（blocks）

线程块的大小和数量是开发者定义的

SM里面会分为若干blocks，但是一个blocks不能同时存在于不同SM里面

通常一个线程块内所有warp和threads会共有一块共享内存，这就有一个平衡点：

$$
单个block的共享内存 \times NUM_{block} < SM的共享内存总量
$$

## 线程束(warp)

$warp \ size$是固定的，通常为32，但是在$K100$里面是64

所以$warp \ num$在$blockDim$和$blockNum$设定好的那一刻就确定好了

$warp$是线程的一个行动小分队，通常会一起出动，这个可以在连续内存访问等地方运用到

## 线程（thread）

在ONNXRuntime中，运行时每一个$thread$都会运行一次算子，而算子的GPU内核代码中的运行时代码会每次都运行一次。

且寄存器是每个线程独有的。

# 编译时和运行时

## 编译时

**定义**：编译时是指程序源代码被编译器处理、分析和优化的阶段，发生在程序实际运行之前。

### 特点
- **静态性**：在程序运行前就确定
- **一次性**：编译完成后不会改变
- **可预测性**：编译器可以完全分析和优化
- **离线处理**：不依赖实际运行环境

### 编译时的主要任务

```cpp
// 编译时确定的内容
const int BLOCK_SIZE = 16;        // 常量定义
constexpr int WARP_SIZE = 32;     // 编译时常量

// 编译时类型检查
template<typename T>
void process_data(T* data) {      // 模板实例化
    // 编译时确定类型T
}

// 编译时内存布局
struct Tensor {
    float* data;    // 编译时确定结构
    int shape[4];   // 编译时确定大小
};
```

### 编译时优化

```cpp
// 编译时循环展开
__global__ void kernel() {
    // 编译器会将这个循环展开
    for (int i = 0; i < 4; i++) {  // 编译时确定循环次数
        // 循环体
    }
}

// 编译时常量折叠
const float PI = 3.14159f;
float result = PI * 2.0f;  // 编译时计算为 6.28318f
```

## 运行时

**定义**：运行时是指程序实际执行、运行和处理的阶段，发生在程序被加载到内存并开始执行之后。

### 运行时的特点

- **动态性**：在程序运行过程中确定
- **实时性**：需要立即响应和处理
- **环境依赖**：依赖运行时的硬件和软件环境
- **在线处理**：需要处理实际的数据和状态

### 运行时的主要任务

```cpp
// 运行时动态分配
void runtime_allocation() {
    int size;
    std::cin >> size;  // 运行时输入
    
    float* data = new float[size];  // 运行时分配内存
    // ... 使用data
    delete[] data;  // 运行时释放内存
}

// 运行时条件判断
void runtime_decision() {
    if (user_input == "GPU") {      // 运行时条件
        use_gpu_kernel();
    } else {
        use_cpu_kernel();
    }
}
```

### 运行时优化

```cpp
// 运行时内存管理
void runtime_memory_management() {
    // 运行时分配GPU内存
    float* gpu_data;
    hipMalloc(&gpu_data, size);
    
    // 运行时异步传输
    hipMemcpyAsync(gpu_data, cpu_data, size, hipMemcpyHostToDevice, stream);
    
    // 运行时同步
    hipStreamSynchronize(stream);
    
    // 运行时释放
    hipFree(gpu_data);
}
```

# 编译时和运行时与GPU线程

## 编译时在GPU线程结构中的体现

### 线程块大小确定（编译时）

```cpp
// 编译时确定线程块大小
dim3 blockDim(16, 16);        // 编译时确定：16x16=256个线程
dim3 gridDim(100, 100);       // 编译时确定：100x100个线程块

// 编译器在编译时就确定了：
// - 每个线程块包含256个线程
// - 网格包含10,000个线程块
// - 总线程数 = 256 × 10,000 = 2,560,000个线程
```

### 线程束数量计算（编译时）

```cpp
// 根据您的资料：warp_num在blockDim和blockNum设定好的那一刻就确定好了
constexpr int WARP_SIZE = 32;  // 编译时常量

// 编译时计算线程束数量
int warp_num_per_block = (blockDim.x * blockDim.y + WARP_SIZE - 1) / WARP_SIZE;
// 对于16x16的线程块：warp_num_per_block = (256 + 31) / 32 = 8个warp

// 编译时确定总线程束数
int total_warp_num = warp_num_per_block * gridDim.x * gridDim.y;
// 总线程束数 = 8 × 100 × 100 = 80,000个warp
```

### 内存布局确定（编译时）

```cpp
// 编译时确定内存布局
struct ThreadBlockLayout {
    // 编译时确定共享内存大小
    static constexpr int SHARED_MEM_SIZE = 16 * 1024;  // 16KB
    
    // 编译时确定寄存器分配
    static constexpr int REGISTERS_PER_THREAD = 32;
    
    // 编译时确定线程索引计算方式
    static constexpr int THREADS_PER_BLOCK = 256;
};

// 编译时优化内存访问模式
__global__ void kernel() {
    // 编译时确定线程索引计算
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // 编译器优化：连续内存访问模式
    data[idx] = input[idx] * 2.0f;
}
```

## 运行时在GPU线程结构中的体现

### 动态网格大小计算（运行时）

```cpp
// 运行时动态计算网格大小
void launch_kernel(int batch_size, int sequence_length) {
    // 编译时确定的线程块大小
    dim3 blockDim(16, 16);  // 编译时确定
    
    // 运行时计算网格大小
    dim3 gridDim((sequence_length + 15) / 16, 
                 (batch_size + 15) / 16);
    
    // 运行时启动内核
    kernel<<<gridDim, blockDim>>>(batch_size, sequence_length);
}

// 运行时示例：
// batch_size = 128, sequence_length = 512
// gridDim = ((512+15)/16, (128+15)/16) = (32, 8)
// 总线程块数 = 32 × 8 = 256个线程块
```

### 运行时线程执行

```cpp
// 根据您的资料：运行时每一个thread都会运行一次算子
__global__ void attention_kernel(int B, int S, int H,
                                 const float* Q,
                                 const float* K, 
                                 const float* V,
                                 float* output) {
    
    // 运行时计算线程索引
    int b = blockIdx.z;                    // 运行时确定：Batch维度
    int i = blockIdx.y * blockDim.y + threadIdx.y;  // 运行时确定：Sequence维度
    int j = blockIdx.x * blockDim.x + threadIdx.x;  // 运行时确定：Hidden维度
    
    // 运行时边界检查
    if (b >= B || i >= S || j >= H) return;
    
    // 运行时执行算子逻辑
    float result = 0.0f;
    for (int k = 0; k < S; ++k) {
        // 运行时计算注意力分数
        float dot = 0.0f;
        for (int h = 0; h < H; ++h) {
            int idx1 = (b * S + i) * H + h;  // 运行时计算索引
            int idx2 = (b * S + k) * H + h;  // 运行时计算索引
            dot += Q[idx1] * K[idx2];
        }
        result += dot * V[(b * S + k) * H + j];
    }
    
    // 运行时写入结果
    int out_idx = (b * S + i) * H + j;  // 运行时计算输出索引
    output[out_idx] = result;
}
```

### 运行时内存管理

```cpp
// 运行时GPU内存管理
void runtime_gpu_operations(int B, int S, int H) {
    // 运行时计算数据大小
    size_t size = B * S * H * sizeof(float);
    
    // 运行时分配GPU内存
    float *d_Q, *d_K, *d_V, *d_Out;
    hipMalloc(&d_Q, size);  // 运行时分配
    hipMalloc(&d_K, size);  // 运行时分配
    hipMalloc(&d_V, size);  // 运行时分配
    hipMalloc(&d_Out, size); // 运行时分配
    
    // 运行时异步传输
    hipStream_t stream;
    hipStreamCreate(&stream);  // 运行时创建流
    
    // 运行时启动内核
    dim3 blockDim(16, 16);  // 编译时确定
    dim3 gridDim((H + 15) / 16, (S + 15) / 16, B);  // 运行时计算
    
    attention_kernel<<<gridDim, blockDim, 0, stream>>>(
        B, S, H, d_Q, d_K, d_V, d_Out);
    
    // 运行时同步
    hipStreamSynchronize(stream);  // 运行时同步
    
    // 运行时释放内存
    hipFree(d_Q); hipFree(d_K); hipFree(d_V); hipFree(d_Out);
    hipStreamDestroy(stream);
}
```

## 编译时与运行时的协同体现

### 线程束协同执行

```cpp
// 编译时确定warp大小，运行时协同执行
__global__ void warp_cooperative_kernel() {
    // 编译时确定warp大小
    constexpr int WARP_SIZE = 32;
    
    // 运行时获取warp内线程ID
    int lane_id = threadIdx.x % WARP_SIZE;  // 运行时确定
    
    // 运行时warp内协同操作
    float val = input[threadIdx.x];
    
    // 运行时warp内求和
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_down(val, offset);  // 运行时warp内通信
    }
    
    // 只有第一个线程写入结果
    if (lane_id == 0) {
        output[blockIdx.x] = val;  // 运行时条件执行
    }
}
```

### 共享内存协同

```cpp
// 编译时确定共享内存大小，运行时协同使用
__global__ void shared_memory_kernel() {
    // 编译时确定共享内存大小
    __shared__ float shared_data[256];  // 编译时确定
    
    // 运行时线程索引
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // 运行时加载数据到共享内存
    shared_data[threadIdx.x] = input[idx];
    __syncthreads();  // 运行时同步
    
    // 运行时使用共享内存计算
    float result = 0.0f;
    for (int i = 0; i < 256; ++i) {
        result += shared_data[i];  // 运行时访问共享内存
    }
    
    // 运行时写入结果
    output[idx] = result;
}
```