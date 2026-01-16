***TPU-MLIR-basic篇（3）***

# 摘要

# 目录

---

# 前言

本人正在搞AI编译器，这个博客大家可以当作学习笔记

# Lowering in TPU-MLIR

模型的转换实际上就是算子的转换

## TopToTpu

### 基本过程

top.Op可以分为两种：`F32`、`INT8`。

- 其中F32可以直接转换为Tpu层的F32、F16、BF16算子
- INT8则需要Calibration(校准量化)才能转换成Tpu.Op

### 混合精度

前后精度不同时，为保证运算时精度相同，TPU-MLIR会在Op之间插入CastOp对算子的输出类型进行转换。

### 具体实现

#### 定义

1. Tpu Dialect + 算子定义

    ```sh
    /tpu-mlir/include/tpu_mlir/Dialect/Tpu/IR/TpuOps.td
    ```

2. 转换Pass定义

   ```sh
   /tpu-mlir/include/tpu_mlir/Conversion/Passes.td
   ```

#### Patterns实现

```sh
/tpu-mlir/include/tpu_mlir/Conversion/TopToTpu/TopLowering.h
```

由`OpRewritePattern`派生`TopLowering`可以将算子lowering到某个特定芯片上面实现，如LoweingBM1684、LoweringBM1684X......
> **添加**：*/tpu-mlir/lib/Conversion/TopToTpu/LoweringBM1684X.cpp*
>
> **实现**：*/tpu-mlir/lib/Conversion/TopToTpu/BM1684X*

#### Pass实现

```sh
/tpu-mlir/lib/Conversion/TopToTpu/TopToTpuPass.cpp
```

- **Define Target**：定义和合法的Dialect和Op
- **Add Patterns**：添加每个算子的Rewrite Pattern
- **Apply Conversion**：
  1. 多次迭代遍历模型并将Patterns应用在模型转换中
  2. 根据需要重复添加与应用Patterns的流程
  3. 类型验证与CastOp

# 量化概述

## 为什么需要量化？

将深度学习网络部署到硬件设备时，因大量权重导致内存过大、且深度学习模型计算量庞大，导致模型部署难度上升

解决方案中，除了高性能专用芯片之外，还可能进行模型压缩，也就是量化

## 本质

将范围很大的数值映射到由低位范围较小的数值中。

## 为什么量化会有效？

举几个例子：

### Multiply-Accumulate

乘加，简称 `MAC`，是一个基础运算单元。它的本质是“先做乘法，再做加法”，即：

`MAC(a,b,c)=a×b+c`

循环并行计算直至得到最终输出结果，其成本与数字位数的位数成二次线性关系，所以***低位的数据比高位数据成本低一些***

### Data Transfer

硬件上计算时，需要将数据在**内存**和**处理器**之间传递

***低位数据在这个过程中成本更低，即转移效率更高***

## 类型

**Post-Training Quantization** *VS* **Quantization-Aware Training**

### Post-Training Quantization

*训练后量化*

即直接对预训练后的模型进行量化，无需或仅需少量data，易于实现

### Quantization-Aware Training

*量化感知训练*

即在训练中模拟量化重新训练模型，需要带标签的数据，重新训练之后的网络能够更加适应量化，所以损失的精度会较少

## 量化方案

### Uniform Quantization

均匀量化

分为**对称量化（Symmetric Quantization）**和**非对称量化(Asymmetric Quantization)**

对称量化又分为**有符号量化**和**无符号量化**。

# 量化推导

$$
r = S (q - Z)
$$

- $r$：实数，浮点值。
- $q$：$r$量化后的整数值。
- $S$：`scale`，浮点值
- $Z$：`zero point`或`offset`，整数值。

$$
S = \frac{max-min}{qmax-qmin}
$$

$$
Z = round(-\frac{min}{S}+qmin)
$$

- $max、min$：量化前的数值范围。
- $qmin、qmax$：量化后的数值范围。

## INT8非对称量化

$$
\begin{align}
q &= round(\frac{r}{S})+Z \\
  &= clamp(round(\frac{r}{S})+Z,qmin,qmax) \\
  &= 
\begin{cases}
qmin, & q < qmin \\
q, & qmin \leq q \leq qmax \\
qmax, & q > qmax
\end{cases}
\end{align}
$$

- $x$：$round(\frac{r}{S})+Z$
- **Sighed INT8**：$qmin=-2^{8-1}$，$qmax=2^{8-1}-1$
- **Unsigned INT8**：$qmin=0$，$qmax=2^8-1$

## INT8对称量化

***与非对称量化唯一的区别就是$Z$限制为0。***

## 对称量化 VS 非对称量化

### 对称量化

- **Signed INT8**：前后的值域（-128~127）都是关于零点对称的，导致可能有一部分值域meaningless。
- **Unsigned INT8**：值域为0～255

$$
S=
\begin{cases}
\frac{thresh}{128}， &int8 \\
\frac{thresh}{255}， &uint8
\end{cases}
$$
$$
tensor \in [-thresh, thresh]
$$
> thresh指的是实际值的绝对最大值

### 非对称量化

可以根据$Z$来拥有动态的映射范围，所以不会有meaningless问题

但是多出的$Z$会让计算更为复杂。

## 卷积/乘法

卷积操作：

$$
\begin{align}
q_y &=\frac{S_wS_x}{S_y}(q_wq_x+\frac{B}{S_wS_x}) \\
    &=M(q_wq_x+\frac{B}{S_wS_x}) \\
    &=2^{-n}M_0(q_wq_x+\frac{B}{S_wS_x})
\end{align}
$$

所以，可以看作一个很大的**整数值**$M_0(q_wq_x+\frac{B}{S_wS_x})$，右移$rshift$位。而此处的$M_0$又称为$Multiplier$
> $Multiplier$因硬件而不同，通常为i32、i16、i8的整数值，位数越高，误差越小。

## 加法运算

原公式：
$$
Y=A+B
$$

量化后：
$$
\begin{align}
&S_y(q_y-Z_y) =S_a(q_a-Z_a)+S_b(q_b-Z_b) \\
&q_y=\frac{S_a}{S_y}(q_a-Z_a)+\frac{S_b}{S_y}(q_b-Z_b)+Z_y
\end{align}
$$

**问题：** 运算次数过多，性能损耗较大。

**解决方案：** 

- *反量化 + 重量化*
  $$
  q_y=requant(dequant(q_a)+dequant(q_b))
  $$
  
- *要求$S_a=S_b$*
  $$
  q_y=\frac{S_a}{S_y}(q_a+q_b-Z_a-Z_b)+Z_y
  $$
  > 八次运算减为六次

## 其他算子的量化

*例如：ReLu、Pad......*

输入输出数据流中绝大部分元素不变。

保证输入数据流的值域相同的话$S$和$Z$也会相同。

## 量化粒度

### Per Tensor

为每个数据流提供统一的量化参数$Scale$和$Zero point$
> 有时过于粗糙

### Per Channel

为每个或几个Channel使用一对$Scale$和$Zero point$

# 量化校准

## 量化门限确定

*怎么确定max、min、threshold?*

## 量化校准(Calibration)

即根据activation数据分布，通过特定算法，来确定量化参数。

**问题：** 大部分模型时不带有数据分布信息的

**解决方案：** 通过筛选+预处理数据，再进行模型推理得到数据分布信息。
> 这就是为什么即便**PTQ**无需重新训练模型也可能需要少量数据
>
> TPU-MLIR提供`run_calibration.py`的接口

## 校准目标

权衡***rounding error***和***clipping error***。

- **Clipping Error**：转换过程中部分值可能会在门限之外。
- **Rounding Error**：转换后与原数值的偏差。

## 校准策略

### Min-Max

直接取绝对最大与最小值为门限。

**优势：**

- 保证*0 clipping error*。
- 当数值均匀分布时较为理想。

**缺陷：**

- 部分离群点会导致性能浪费
- 会增多*Rounding Error*。

### KLD

#### KL散度

$$
D_{KL}(P||Q) =
\begin{cases}
\sum_i P(i) \log \frac{P(i)}{Q(i)}, &离散 \\\\
\int_{-\infty}^{\infty} p(x) \log \frac{p(x)}{q(x)}, &连续
\end{cases}
$$

即用来评判两个量化分布的的相似程度，**KL散度越大，分布差异越大。**

### KLD策略

找到最优的量化门限，使得量化后的数据分布与原始数据分布最相似。

用2048bins的直方图来表示FP32的数据分布，每格迭代128bins的截取来获取最优门限

**问题：**

- 粒度较粗
- 不同场景下tensor中各个元素的重要性不一样。

### In TPU-MLIR

在KLD得出的最优门限和Min-Max得出的最优门限之间切分出若干Candidate，然后经过反量化和重量化之后计算欧式距离

最终选取欧式距离最小的。

# 量化感知训练