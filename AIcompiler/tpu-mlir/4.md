**AI编译器-tpu_mlir篇（4）**

# 摘要

# 目录

# 前言

本人正在搞AI编译器，这个博客大家可以当作学习笔记

# 精度验证

## Why?

在模型转换过程中，我们需评估模型的**Similarity**来确保模型转换的正确性。

## 数据集准备

- **分类网络**：https://www.image-net.org/challenges/LSVRC/2012/
  > 1000 subdir -> 50 images
- **目标检测网络**：https://cocodataset.org/#download
  > 放在一个目录下，每张图片信息都会从对应的annotation文件中提取。

## 接口使用

### model_eval.py接口

TPU-MLIR提供了`model_eval.py`接口用于模型精度验证，支持多种数据集类型和后处理方式。

#### 基本语法
```bash
python3 -m tpu_mlir.tools.model_eval \
    --model_file <model_file> \
    --dataset_type <dataset_type> \
    --postprocess_type <postprocess_type> \
    --dataset <dataset_path> \
    --count <image_count>
```

#### 主要参数

| 参数名 | 类型 | 必需 | 说明 | 示例 |
|--------|------|------|------|------|
| `--model_file` | str | 是 | 模型文件路径 | `mobilenet_v2.mlir` |
| `--dataset_type` | str | 否 | 数据集类型 | `imagenet`, `coco`, `voc`, `user_define` |
| `--postprocess_type` | str | 是 | 后处理类型 | `topx`, `coco_mAP` |
| `--dataset` | str | 否 | 数据集路径 | `datasets/ILSVRC2012_img_val_with_subdir` |
| `--count` | int | 否 | 验证图片数量 | `50000` |
| `--label_file` | str | 否 | 标签文件 | `labels.txt` |
| `--coco_annotation` | str | 否 | COCO标注文件 | `instances_val2017.json` |
| `--debug_cmd` | str | 否 | 调试命令 | `not_use_preprocess` |

#### 使用示例

##### 1. ImageNet分类模型评估
```bash
# F32模型评估
python3 -m tpu_mlir.tools.model_eval \
    --model_file mobilenet_v2.mlir \
    --count 50000 \
    --dataset_type imagenet \
    --postprocess_type topx \
    --dataset datasets/ILSVRC2012_img_val_with_subdir

# INT8量化模型评估
python3 -m tpu_mlir.tools.model_eval \
    --model_file mobilenet_v2_bm1684x_int8_sym_tpu.mlir \
    --count 50000 \
    --dataset_type imagenet \
    --postprocess_type topx \
    --dataset datasets/ILSVRC2012_img_val_with_subdir
```

##### 2. COCO目标检测模型评估
```bash
# F32模型评估
python3 -m tpu_mlir.tools.model_eval \
    --model_file yolov5s.mlir \
    --count 5000 \
    --dataset_type coco \
    --postprocess_type coco_mAP \
    --coco_annotation datasets/instances_val2017.json \
    --dataset datasets/val2017

# INT8量化模型评估
python3 -m tpu_mlir.tools.model_eval \
    --model_file yolov5s_bm1684x_int8_sym_tpu.mlir \
    --count 5000 \
    --dataset_type coco \
    --postprocess_type coco_mAP \
    --coco_annotation datasets/instances_val2017.json \
    --dataset datasets/val2017
```

##### 3. 自定义数据集评估
```bash
python3 -m tpu_mlir.tools.model_eval \
    --model_file model.mlir \
    --dataset_type user_define \
    --postprocess_type topx \
    --dataset /path/to/images \
    --data_list image_list.txt \
    --count 1000
```

#### 输出结果示例

##### 分类任务结果
```
2022/11/08 01:30:29 - INFO : idx:50000, top1:0.710, top5:0.899
INFO:root:idx:50000, top1:0.710, top5:0.899
```

##### 检测任务结果
```
Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.369
Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.561
Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.393
```

#### 调试选项

```bash
# 不使用预处理
--debug_cmd not_use_preprocess

# 指定批次大小
--debug_cmd batch_size=4

# 启用详细日志
--debug_cmd debug_log
```

## 预处理

***为什么不需要预处理参数？***

在MLIR的InputOp里，就包含了预处理信息。

### MeanStdScale

TPU-MLIR中的预处理算子，实现标准化和缩放:

***commonly used：***

$$
(\frac{x}{255}-mean)/std \Rightarrow (x-255 \times mean)/255 \times std
$$

- **$mean$**: 数据中心化的均值，用于标准化。
- **$std$**：数据标准化的标准差

***In TPU-MLIR***

$$
(x-mean_l) \times scale_l
\Rightarrow
\begin{cases}
mean_l=255 \times mean \\\\
scale_l=\frac{1}{255 \times std}
\end{cases}
$$

## Metrics

### 分类网络

**TopX**：模型输出的概率中排序最高(Top1)和前五(Top5)的类别中是否包含了正确的类别。

### 目标检测网络

COCO官网提供的12个指标。

# Pattern Rewritting

前面曾说过，Rewrite Pattern时实现了非法Op转换到合法Op

***实际上就是DAG-to-DAG transformation***

包含三个部分：

- **Pattern Definition**
- **Pattern Rewriter**
- **Pattern Application**

## Pattern Definition

所有**Pattern**均继承自`RewritePattern`这个父类