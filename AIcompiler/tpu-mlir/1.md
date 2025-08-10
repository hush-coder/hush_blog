**AI编译器-tpu_mlir篇（1）**

# 摘要

本篇笔记系统梳理了AI编译器tpu_mlir的整体架构、核心概念与关键实现，包括模型转换流程、MLIR语法、类型系统、Operation、Attribute等基础知识，并结合深度学习实际场景给出丰富实例，帮助读者快速理解AI编译器的原理与工程实现。

# 目录

- [摘要](#摘要)
- [目录](#目录)
- [前言](#前言)
- [AI编译器](#ai编译器)
- [整体架构](#整体架构)
  - [流程](#流程)
  - [python接口](#python接口)
    - [MODEL to MLIR](#model-to-mlir)
    - [MLIR to F16 bmodel](#mlir-to-f16-bmodel)
    - [MLIR to INT8 bmodel](#mlir-to-int8-bmodel)
- [一些基础](#一些基础)
  - [什么是张量？](#什么是张量)
  - [什么是算子？](#什么是算子)
- [MLIR语法](#mlir语法)
  - [一些问题](#一些问题)
  - [Dialect](#dialect)
    - [定义](#定义)
    - [几个名词](#几个名词)
    - [整体实现方式](#整体实现方式)
  - [基本概念](#基本概念)
  - [Operation](#operation)
  - [Value](#value)
  - [Type](#type)
    - [mlir::Type (基类)](#mlirtype-基类)
    - [ShapedType (有形状的类型)](#shapedtype-有形状的类型)
    - [TensorType (张量类型)](#tensortype-张量类型)
    - [实际使用示例](#实际使用示例)
    - [在深度学习中的意义](#在深度学习中的意义)
  - [Attribute](#attribute)
    - [定义](#定义-1)
    - [Attribute 的作用](#attribute-的作用)
    - [Attribute 与 Value 的区别](#attribute-与-value-的区别)
    - [常见的MLIR Attribute类型](#常见的mlir-attribute类型)
      - [1. 数值Attribute](#1-数值attribute)
      - [2. 张量Attribute](#2-张量attribute)
      - [3. 字符串Attribute](#3-字符串attribute)
      - [4. 数组Attribute](#4-数组attribute)
    - [在深度学习中的常见Attribute](#在深度学习中的常见attribute)
      - [卷积操作](#卷积操作)
      - [池化操作](#池化操作)
    - [Attribute 的访问方式](#attribute-的访问方式)
      - [C++代码中访问Attribute](#c代码中访问attribute)
    - [Attribute 的重要性](#attribute-的重要性)
    - [在TPU-MLIR中的意义](#在tpu-mlir中的意义)
    - [具体实例](#具体实例)
      - [实例1：常量操作](#实例1常量操作)
      - [实例2：卷积操作](#实例2卷积操作)
      - [实例3：池化操作](#实例3池化操作)
      - [实例4：量化相关Attribute](#实例4量化相关attribute)
      - [实例5：形状变换操作](#实例5形状变换操作)
      - [实例6：激活函数](#实例6激活函数)
    - [实际应用场景](#实际应用场景)
      - [场景1：模型转换时的Attribute处理](#场景1模型转换时的attribute处理)
      - [场景2：量化时的Attribute设置](#场景2量化时的attribute设置)
      - [场景3：硬件特定的Attribute](#场景3硬件特定的attribute)


# 前言

本人正在搞AI编译器，这个博客大家可以当作学习笔记

# AI编译器

压榨人工智能的性能

**TPU**：专用于加速深层神经网络运算的定制化ASIC芯片。

**定义**：将不同框架下的搭建起来的模型转化为统一的中间表达（MLIR），转化为某一平台上的代码。

中间表达又被分为**芯片无关层（Top）**和**芯片有关层（Tpu）**

> **Top：** 图优化、量化...
>
> **Tpu：** 权重重排、算子切分、地址分配...

# 整体架构

## 流程

1. 由**ONNX**的算子按顺序一一对应进行转换生成`origin.mlir`
2.  `origin.mlir`通过图优化生成`canonical.mlir`(**Top**层)
3.  转向**Tpu**层：
   - 若是F32/BF16精度的转换，可直接将Top层的mlir下降到**Tpu**层，生成`tpu.mlir`
   - 若是INT8类型的转换，需先通过calibration建立校准表`cali_table`，再结合该量化表将mlir模型lower到**Tpu**层
4. 在tpu层中继续进行优化操作，最后生成一个可以在tpu上可以直接运行的二进制model.bmodel文件。
5. 模型转换过程中，会在各个阶段进行模型推理，并对比他们的结果，以确保转换的正确性。

## python接口

以yolov5s.onnx为例

### MODEL to MLIR

```sh
model_transform.py \
    --model_name yolov5s \
    --model_def ../yolov5s.onnx \
    --input_shapes [[1,3,640,640]] \
    --mean 0.0,0.0,0.0 \
    --scale 0.0039216,0.0039216,0.0039216 \
    --keep_aspect_ratio \
    --pixel_format rgb \
    --output_names 350,498,646 \
    --test_input ../image/dog.jpg \
    --test_result yolov5s_top_outputs.npz \
    --mlir yolov5s.mlir
```

### MLIR to F16 bmodel

```sh
model_deploy.py \
  --mlir yolov5s.mlir \
  --quantize F16 \
  --processor bm1684x \
  --test_input yolov5s_in_f32.npz \
  --test_reference yolov5s_top_outputs.npz \
  --model yolov5s_1684x_f16.bmodel
```

### MLIR to INT8 bmodel

生成校准表

```sh
run_calibration.py yolov5s.mlir \
  --dataset ../COCO2017 \
  --input_num 100 \
  -o yolov5s_cali_table
```

生成bmodel模型

```sh
model_deploy.py \
  --mlir yolov5s.mlir \
  --quantize INT8 \
  --calibration_table yolov5s_cali_table \
  --processor bm1684x \
  --test_input yolov5s_in_f32.npz \
  --test_reference yolov5s_top_outputs.npz \**整数值**
  --tolerance 0.85,0.45 \
  --model yolov5s_1684x_int8.bmodel
```

# 一些基础

## 什么是张量？

- **定义：** 张量是深度学习框架中用于表示数据的基本数据结构，可以将其看作多维数组。
- **维度：**
  - 0维张量：标量
  - 1维张量：向量
  - 2维张量：矩阵
  - n维张量

## 什么是算子？

- **定义：** 也成操作符、层或函数，是深度学习计算图中执行计算的基本单元。定义了如何将一个的或多个输入张量转换为输出张量。
- **在计算图中的角色：** 算子构成了计算图的节点。


# MLIR语法

## 一些问题

- **什么是MLIR?** 
  
  *一个LLVM的子项目，用于模糊化不同IR的界限*
- **为什么需要多层IR？** 

  *IR越高级，其提取源码意图难度越低，针对特定硬件优化难度越高，则单层或少数层IR难以实现。*
- **什么是dyn_cast？**
  
  *`dyn_cast` 是LLVM中的一个**安全的向下类型转换**（downcast）工具，用于将基类指针转换为派生类指针。可以保证避免空指针问题以及目标派生类不存在的危险转换问题*


## Dialect

### 定义

不同层级的Dialect不断抽象，最后形成对于特定硬件优化的代码。

### 几个名词

- **Prefix:** 命名空间，即这个Dialect的名称.。
- **Operations:** 一系列操作，每个操作对应深度学习模型中的某个算子。
- **Passes:** 包括Dialect内的转换、Dialect间的转换

### 整体实现方式

1. 后端对每个**Dialect**进行初始化
2. 用**TableGen**工具根据相应Dialect定义算子的.td文件，自动生成对应的C++文件，再实现每个算子的运算逻辑
3. 对**getDialect**文件的优化操作以及向下抽象操作进行实现
4. 创建python接口，用于解析并转换源模型为最底层Dialect的mlir文件

## 基本概念

一个完整的MLIR文本有三个部分组成：ModuleOp,FuncOp,Block，他们逐级包含

- **ModuleOp：** 代表当前IR所表示的代码本身
- **FuncOp：** 一个Module中可以有多个function（必有一个main function）
- **Block：** 一切操作的集合

还有一些基础概念，可以提前了解一下：

- **Operation：** 运算
- **Value：** 操作数
- **Type：** Value的类型
- **Attribute：** Op的属性

## Operation

MLIR中的一个**基本执行单元**，在特定大类中对源码或上一层进行表达或者优化，由*Operation类*和*Op类*组成

- **Operation类：** 通用定义，提供通用接口和属性
- **Op类：** 各种特定定义Operation的基类。
- `ConstantOp op`通过`op.getOperation()`到`mlir:Operation* operation`，相反则是`llvm::dyn_cast<ConstantOp>(operation)`。

## Value

主要有两个派生类

- **BlockArgument：** 某个Block的输入参数
- **OpResult：** 以静态单赋值的形式存储每个Op的结果。
  > 静态单赋值SSA是IR中的属性，要求每个变量值分配一次，并在每个变量使用之前定义。
  >> 通过该特性，我们可以很方便的获取其Op的特性：`value.getDefiningOp()`(通用)或`value.getDefiningOp<toy::MulOp>()`（特定）
  >> 同时，Operation可以通过`operation->getResult(0)`(此处的0是index标识，因为一个op可能有很多个value)获取其value。

## Type

分为mlir::Type、ShapedType、TensorType、具体张量类型。逐级继承。

### mlir::Type (基类)

- **定义：** 所有MLIR类型的基类
- **特点：**
  - 最抽象的类型
  - 不包含具体的数据形状信息
  - 提供基本的类型功能

```cpp
// 基本用法
mlir::Type type = someValue.getType();
bool isFloat = type.isF32();  // 检查是否是f32类型
```

### ShapedType (有形状的类型)

- **定义：** 具有形状信息的类型基类
- **特点：**
  - 包含维度信息
  - 有元素类型
  - 支持形状查询
- **包含的类型：**
  - TensorType (张量)
  - VectorType (向量)
  - MemRefType (内存引用)

```cpp
// 检查是否是ShapedType
if (auto shapedType = type.dyn_cast<mlir::ShapedType>()) {
    // 获取形状信息
    auto shape = shapedType.getShape();  // 返回维度数组
    auto elementType = shapedType.getElementType();  // 获取元素类型
    auto rank = shapedType.getRank();  // 获取维度数
}
```

### TensorType (张量类型)

- **定义：** 表示多维数组的类型
- **特点：**
  - 继承自ShapedType
  - 专门用于表示张量数据
  - 在深度学习中非常常用

```cpp
// 检查是否是TensorType
if (auto tensorType = type.dyn_cast<mlir::TensorType>()) {
    // 获取张量特有的信息
    auto shape = tensorType.getShape();
    auto elementType = tensorType.getElementType();
    
    // 检查是否是静态形状
    bool isStatic = tensorType.hasStaticShape();
}
```

### 实际使用示例

```cpp
// 假设我们有一个Value
mlir::Value value = someOperation->getResult(0);
mlir::Type type = value.getType();

// 层次化的类型检查
if (auto shapedType = type.dyn_cast<mlir::ShapedType>()) {
    // 这是一个有形状的类型
    llvm::ArrayRef<int64_t> shape = shapedType.getShape();
    
    if (auto tensorType = type.dyn_cast<mlir::TensorType>()) {
        // 这是一个张量类型
        if (tensorType.hasStaticShape()) {
            // 静态形状张量
            for (int64_t dim : shape) {
                // 处理每个维度
            }
        }
    }
}
```

### 在深度学习中的意义

1. **mlir::Type**：提供基本的类型检查功能
2. **ShapedType**：处理所有有形状的数据（张量、向量等）
3. **TensorType**：专门处理深度学习中的张量数据

这种层次设计让MLIR能够：

- 统一处理不同类型的有形状数据
- 提供类型安全的操作
- 支持复杂的形状推理和优化

## Attribute

### 定义

Attribute 是 **Op的属性**，用于存储Operation的**静态信息**和**配置参数**。
> Attribute对于属性，就像type对于value
>> 属性的类型有UI64Attr、F32Attr等等

### Attribute 的作用

Attribute告诉系统：

- Operation的**配置参数**（如卷积的stride、padding）
- **常量值**（如权重、偏置）
- **元数据信息**（如数据类型、形状信息）
- **编译时**确定的参数

### Attribute 与 Value 的区别

| 特性 | Attribute | Value |
|------|-----------|-------|
| **时机** | 编译时确定 | 运行时确定 |
| **存储** | 静态信息 | 动态数据 |
| **类型** | 配置参数 | 计算数据 |
| **例子** | 卷积的stride=2 | 输入张量数据 |

### 常见的MLIR Attribute类型

#### 1. 数值Attribute

```mlir
// 整数Attribute
%0 = "toy.constant"() {value = 42 : i32} : () -> i32

// 浮点Attribute
%1 = "toy.constant"() {value = 3.14 : f32} : () -> f32
```

#### 2. 张量Attribute

```mlir
// 常量张量
%2 = "toy.constant"() {
  value = dense<[[1, 2], [3, 4]]> : tensor<2x2xi32>
} : () -> tensor<2x2xi32>

// 权重张量
%3 = "conv.weight"() {
  weight = dense<[[1, 0], [0, 1]]> : tensor<2x2xf32>
} : () -> tensor<2x2xf32>
```

#### 3. 字符串Attribute

```mlir
// 操作名称
%4 = "toy.operation"() {
  name = "conv2d" : string
} : () -> tensor<1x3x224x224xf32>
```

#### 4. 数组Attribute

```mlir
// 形状信息
%5 = "toy.reshape"(%input) {
  shape = [1, 3, 224, 224] : tensor<4xi64>
} : (tensor<*xf32>) -> tensor<1x3x224x224xf32>
```

### 在深度学习中的常见Attribute

#### 卷积操作

```mlir
%conv = "tpu.conv2d"(%input, %weight) {
  strides = [1, 1],           // 步长
  padding = [1, 1, 1, 1],     // 填充
  dilation = [1, 1],          // 膨胀
  groups = 1                   // 分组数
} : (tensor<1x3x224x224xf32>, tensor<64x3x3x3xf32>) -> tensor<1x64x224x224xf32>
```

#### 池化操作

```mlir
%pool = "tpu.maxpool2d"(%input) {
  kernel_shape = [2, 2],      // 池化核大小
  strides = [2, 2],           // 步长
  padding = [0, 0, 0, 0]      // 填充
} : (tensor<1x64x224x224xf32>) -> tensor<1x64x112x112xf32>
```

### Attribute 的访问方式

#### C++代码中访问Attribute

```cpp
// 获取Operation
mlir::Operation* op = someValue.getDefiningOp();

// 获取特定Attribute
if (auto strideAttr = op->getAttr("strides")) {
    if (auto arrayAttr = strideAttr.dyn_cast<mlir::ArrayAttr>()) {
        // 访问数组Attribute
        for (auto attr : arrayAttr) {
            int64_t value = attr.cast<mlir::IntegerAttr>().getInt();
        }
    }
}

// 获取数值Attribute
if (auto valueAttr = op->getAttr("value")) {
    if (auto intAttr = valueAttr.dyn_cast<mlir::IntegerAttr>()) {
        int64_t value = intAttr.getInt();
    }
}
```

### Attribute 的重要性

1. **编译时优化**：编译器可以根据Attribute进行优化
2. **硬件适配**：不同硬件支持不同的Attribute参数
3. **代码生成**：根据Attribute生成对应的硬件代码
4. **模型转换**：在不同框架间转换时保持参数信息

### 在TPU-MLIR中的意义

在AI编译器中，Attribute特别重要：

- **量化参数**：scale、zero_point等量化信息
- **算子参数**：卷积的kernel_size、stride等
- **常量数据**：权重、偏置等静态数据
- **配置信息**：数据类型、精度等元信息

Attribute是MLIR中连接**静态配置**和**动态计算**的重要桥梁！

### 具体实例

#### 实例1：常量操作
```mlir
// 定义一个常量，值为42
%const_42 = "toy.constant"() {value = 42 : i32} : () -> i32

// 定义一个浮点常量
%const_pi = "toy.constant"() {value = 3.14159 : f32} : () -> f32

// 定义一个权重张量
%weight = "toy.constant"() {
  value = dense<[[1.0, 0.5], [0.3, 0.8]]> : tensor<2x2xf32>
} : () -> tensor<2x2xf32>
```

#### 实例2：卷积操作
```mlir
// 标准卷积操作
%conv = "tpu.conv2d"(%input, %weight) {
  strides = [1, 1],                    // 步长为1x1
  padding = [1, 1, 1, 1],             // 填充为1
  dilation = [1, 1],                   // 膨胀为1x1
  groups = 1,                          // 分组数为1
  do_relu = true                       // 激活函数
} : (tensor<1x3x224x224xf32>, tensor<64x3x3x3xf32>) -> tensor<1x64x224x224xf32>

// 深度可分离卷积
%depthwise_conv = "tpu.depthwise_conv2d"(%input, %depthwise_weight) {
  strides = [2, 2],                    // 步长为2x2
  padding = [0, 0, 0, 0],             // 无填充
  dilation = [1, 1],                   // 膨胀为1x1
  multiplier = 1                        // 乘数因子
} : (tensor<1x64x112x112xf32>, tensor<64x1x3x3xf32>) -> tensor<1x64x56x56xf32>
```

#### 实例3：池化操作
```mlir
// 最大池化
%maxpool = "tpu.maxpool2d"(%input) {
  kernel_shape = [3, 3],               // 3x3池化核
  strides = [2, 2],                    // 步长为2x2
  padding = [1, 1, 1, 1],             // 填充为1
  ceil_mode = false                     // 向下取整
} : (tensor<1x64x112x112xf32>) -> tensor<1x64x56x56xf32>

// 平均池化
%avgpool = "tpu.avgpool2d"(%input) {
  kernel_shape = [2, 2],               // 2x2池化核
  strides = [2, 2],                    // 步长为2x2
  padding = [0, 0, 0, 0],             // 无填充
  count_include_pad = false             // 不包含填充
} : (tensor<1x64x56x56xf32>) -> tensor<1x64x28x28xf32>
```

#### 实例4：量化相关Attribute
```mlir
// 量化卷积
%quantized_conv = "tpu.conv2d"(%input, %weight) {
  strides = [1, 1],
  padding = [1, 1, 1, 1],
  input_scale = 0.0078125,             // 输入缩放因子
  weight_scale = 0.00390625,           // 权重缩放因子
  output_scale = 0.015625,             // 输出缩放因子
  input_zero_point = 128,              // 输入零点
  weight_zero_point = 0,               // 权重零点
  output_zero_point = 128              // 输出零点
} : (tensor<1x3x224x224xi8>, tensor<64x3x3x3xi8>) -> tensor<1x64x224x224xi8>
```

#### 实例5：形状变换操作
```mlir
// 重塑操作
%reshape = "tpu.reshape"(%input) {
  shape = [1, 3, 224, 224] : tensor<4xi64>  // 目标形状
} : (tensor<*xf32>) -> tensor<1x3x224x224xf32>

// 转置操作
%transpose = "tpu.transpose"(%input) {
  perm = [0, 2, 3, 1] : tensor<4xi64>       // 维度排列
} : (tensor<1x3x224x224xf32>) -> tensor<1x224x224x3xf32>
```

#### 实例6：激活函数
```mlir
// ReLU激活
%relu = "tpu.relu"(%input) {
  slope = 0.0 : f32                    // 斜率（ReLU为0）
} : (tensor<1x64x224x224xf32>) -> tensor<1x64x224x224xf32>

// Leaky ReLU
%leaky_relu = "tpu.relu"(%input) {
  slope = 0.1 : f32                    // 斜率为0.1
} : (tensor<1x64x224x224xf32>) -> tensor<1x64x224x224xf32>
```

### 实际应用场景

#### 场景1：模型转换时的Attribute处理
```cpp
// 从ONNX转换到MLIR时，需要提取Attribute
void convertConv2d(onnx::NodeProto* onnx_node, mlir::OpBuilder& builder) {
    // 提取strides Attribute
    std::vector<int64_t> strides;
    if (auto strides_attr = onnx_node->getAttribute("strides")) {
        for (auto stride : strides_attr->ints()) {
            strides.push_back(stride);
        }
    }
    
    // 提取pads Attribute
    std::vector<int64_t> pads;
    if (auto pads_attr = onnx_node->getAttribute("pads")) {
        for (auto pad : pads_attr->ints()) {
            pads.push_back(pad);
        }
    }
    
    // 创建MLIR操作，包含Attribute
    auto conv_op = builder.create<Conv2dOp>(
        location,
        input,
        weight,
        builder.getI64ArrayAttr(strides),    // strides Attribute
        builder.getI64ArrayAttr(pads),       // pads Attribute
        builder.getI64IntegerAttr(1)         // groups Attribute
    );
}
```

#### 场景2：量化时的Attribute设置
```cpp
// 设置量化Attribute
void setQuantizationAttributes(mlir::Operation* op, 
                             float input_scale, 
                             float weight_scale, 
                             float output_scale) {
    // 设置输入量化参数
    op->setAttr("input_scale", 
                mlir::FloatAttr::get(builder.getF32Type(), input_scale));
    
    // 设置权重量化参数
    op->setAttr("weight_scale", 
                mlir::FloatAttr::get(builder.getF32Type(), weight_scale));
    
    // 设置输出量化参数
    op->setAttr("output_scale", 
                mlir::FloatAttr::get(builder.getF32Type(), output_scale));
}
```

#### 场景3：硬件特定的Attribute
```cpp
// 为特定TPU设置Attribute
void setTPUSpecificAttributes(mlir::Operation* op, std::string processor) {
    if (processor == "bm1684x") {
        // BM1684X特定的Attribute
        op->setAttr("chip", mlir::StringAttr::get(context, "bm1684x"));
        op->setAttr("precision", mlir::StringAttr::get(context, "int8"));
    } else if (processor == "bm1688") {
        // BM1688特定的Attribute
        op->setAttr("chip", mlir::StringAttr::get(context, "bm1688"));
        op->setAttr("precision", mlir::StringAttr::get(context, "fp16"));
    }
}
```

这些实例展示了Attribute在MLIR中的实际应用，从基本的常量定义到复杂的深度学习操作，Attribute都扮演着关键角色！