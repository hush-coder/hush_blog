# TPU-MLIR 形状推断（Shape Inference）实现指南

## 1. 问题背景

### 1.1 问题描述
在TPU-MLIR项目开发过程中，我们遇到了以下关键问题：

- **UnrankedTensorType问题**：某些模型层的输出类型为 `<* x dtype>`，类型中缺少秩和维度信息
- **形状信息缺失**：原始模型层的形状信息不正确或不完整
- **编译时形状推断需求**：需要在编译时推断出正确的输出形状以支持后续优化

### 1.2 影响分析
形状信息缺失会导致：
- 编译器无法进行有效的优化
- 内存分配错误
- 运行时形状不匹配错误
- 性能下降

### 1.3 解决方案
实现完整的形状推断机制，在编译时自动推断和验证所有操作的输出形状。

## 2. 技术架构

### 2.1 整体架构
```
输入模型 → 形状推断 → 形状验证 → 优化编译 → 输出代码
    ↓           ↓         ↓         ↓         ↓
  MLIR IR   推断形状   验证正确性   应用优化   生成代码
```

### 2.2 核心组件
- **ShapeInterface.cpp**：通用形状推断函数库
- **Top方言接口**：每个操作的具体形状推断实现
- **形状验证机制**：确保推断结果的正确性

### 2.3 设计原则
1. **统一接口**：所有操作都实现相同的`shape_inference()`方法
2. **分层实现**：通用函数 + 特定操作实现
3. **错误处理**：优雅处理形状推断失败的情况
4. **性能考虑**：避免重复计算，缓存推断结果

## 3. 实现指南

### 3.1 通用形状推断函数

#### 3.1.1 common_shape_inference
适用于输入输出形状相同的操作（如激活函数）：
```cpp
void common_shape_inference(mlir::Operation *op) {
    if (op->getNumResults() != 1) {
        UNREACHABLE_OP("input and output should be only one", op);
    }
    auto in = op->getOperand(0);
    auto out = op->getResult(0);
    auto in_shape = module::getShape(in);
    module::setShapeOrVerify(out, in_shape);
    
    // 处理标量消费者特征
    if (op->hasTrait<trait::ScalarConsumer>()) {
        auto context = op->getContext();
        mlir::Builder builder(context);
        auto pre_op = in.getDefiningOp();
        auto is_scalar = module::isScalar(pre_op);
        op->setAttr("is_scalar", builder.getBoolAttr(is_scalar));
    }
}
```

#### 3.1.2 broadcast_shape_inference
适用于需要广播的操作：
```cpp
void broadcast_shape_inference(mlir::Operation *op) {
    auto out_shape = computer_broadcast_shape(op);
    auto out = op->getResult(0);
    module::setShapeOrVerify(out, out_shape);
}
```

### 3.2 操作特定实现

#### 3.2.1 简单操作实现
对于激活函数、一元操作等简单操作：
```cpp
void top::ReluOp::shape_inference() { 
    common_shape_inference(getOperation()); 
}

void top::TanhOp::shape_inference() { 
    common_shape_inference(getOperation()); 
}

void top::SqrtOp::shape_inference() { 
    common_shape_inference(getOperation()); 
}
```

#### 3.2.2 复杂操作实现
对于卷积、池化等复杂操作：
```cpp
void top::ConvOp::shape_inference() {
    // 获取输入和滤波器形状
    auto input_shape = module::getShape(getInput());
    auto filter_shape = module::getShape(getFilter());
    
    // 验证输入形状
    ASSERT_THIS(input_shape.size() == filter_shape.size());
    ASSERT_THIS(input_shape.size() > 2);
    
    int spacial_rank = input_shape.size() - 2;
    if (spacial_rank != getKernelShape().size()) {
        ASSERT_THIS(module::isUnranked(getOutput()) == false);
        return;
    }
    
    // 计算输出形状
    llvm::SmallVector<int64_t> out_shape;
    out_shape.push_back(input_shape[0]);  // batch size
    out_shape.push_back(filter_shape[0]); // output channels
    
    auto input_spacial_shape = llvm::ArrayRef(&input_shape[2], spacial_rank);
    auto filter_spacial_shape = llvm::ArrayRef(&filter_shape[2], spacial_rank);
    auto pads = module::getI64Array(getPads());
    auto strides = module::getI64Array(getStrides());
    auto dilation = module::getI64Array(getDilations(), spacial_rank, 1);
    
    // 处理auto_pad
    if (getAutoPad().has_value()) {
        auto kernel_shape = module::getI64Array(getKernelShapeAttr());
        std::vector<int64_t> new_pads(pads->begin(), pads->end());
        set_auto_pad(getAutoPad().value(), input_shape, *kernel_shape, *strides, new_pads);
        auto builder = OpBuilder(getContext());
        setPadsAttr(builder.getI64ArrayAttr(new_pads));
        removeAutoPadAttr();
        pads = module::getI64Array(getPads());
    }
    
    // 计算空间维度
    for (int i = 0; i < spacial_rank; i++) {
        auto out_dim = (input_spacial_shape[i] + pads->at(i) + 
                       pads->at(i + spacial_rank) - 
                       dilation->at(i) * (filter_spacial_shape[i] - 1) - 1) / 
                       strides->at(i) + 1;
        out_shape.push_back(out_dim);
    }
    
    // 设置输出形状
    module::setShapeOrVerify(getOutput(), out_shape);
}
```

#### 3.2.3 特殊操作实现
对于需要特殊处理的操作：
```cpp
void top::ReshapeOp::shape_inference() {
    auto input_shape = module::getShape(getInput());
    auto target_shape = module::getI64Array(getTargetShape());
    
    // 处理动态维度
    llvm::SmallVector<int64_t> out_shape;
    int64_t total_elements = 1;
    for (auto dim : input_shape) {
        total_elements *= dim;
    }
    
    int64_t dynamic_count = 0;
    int64_t static_product = 1;
    
    for (auto dim : *target_shape) {
        if (dim == -1) {
            dynamic_count++;
        } else {
            static_product *= dim;
            out_shape.push_back(dim);
        }
    }
    
    // 计算动态维度
    if (dynamic_count == 1) {
        int64_t dynamic_dim = total_elements / static_product;
        // 找到-1的位置并替换
        for (size_t i = 0; i < target_shape->size(); i++) {
            if ((*target_shape)[i] == -1) {
                out_shape.insert(out_shape.begin() + i, dynamic_dim);
                break;
            }
        }
    }
    
    module::setShapeOrVerify(getOutput(), out_shape);
}
```

### 3.3 实现检查清单

每个`shape_inference()`方法必须包含：

1. **输入验证**：检查输入操作数的有效性
2. **形状获取**：获取输入张量的形状信息
3. **形状计算**：根据操作类型计算输出形状
4. **结果设置**：调用`module::setShapeOrVerify()`
5. **错误处理**：处理异常情况

## 4. 测试验证

### 4.1 测试命令
```bash
# 测试单个操作
test_onnx.py --case Conv

# 测试多个操作
test_onnx.py --case Conv,Relu,Tanh

# 运行所有测试
test_onnx.py
```

### 4.2 测试用例要求
每个测试用例应该包含：
- 不同输入形状的组合
- 边界条件测试
- 错误情况处理
- 性能基准测试

### 4.3 验证标准
- 形状推断结果正确
- 与ONNX参考实现一致
- 性能满足要求
- 错误处理正确

## 5. 开发流程

### 5.1 任务分配
1. **分析操作类型**：确定操作的复杂度和实现方式
2. **选择实现策略**：通用函数 vs 自定义实现
3. **实现核心逻辑**：编写形状计算代码
4. **添加测试用例**：确保功能正确性
5. **代码审查**：团队代码审查和优化

### 5.2 提交规范
```bash
# 创建功能分支
git checkout -b feature/shape-inference-{operation_name}

# 实现功能
# ... 编写代码 ...

# 运行测试
test_onnx.py --case {operation_name}

# 提交代码
git add .
git commit -m "feat: implement shape inference for {operation_name}"

# 创建PR
git push origin feature/shape-inference-{operation_name}
```

### 5.3 代码审查要点
- 形状推断逻辑正确性
- 错误处理完整性
- 代码风格一致性
- 性能影响评估
- 测试覆盖充分性

## 6. 常见问题与解决方案

### 6.1 动态形状处理
**问题**：输入包含动态维度（-1）
**解决方案**：
```cpp
// 检查动态维度
if (module::isDynamic(input_shape[i])) {
    // 使用符号推理或延迟推断
    out_shape.push_back(ShapedType::kDynamic);
} else {
    // 正常计算
    out_shape.push_back(computed_dim);
}
```

### 6.2 形状不匹配错误
**问题**：推断形状与预期形状不一致
**解决方案**：
```cpp
// 使用setShapeOrVerify进行验证
module::setShapeOrVerify(out, computed_shape);
// 如果验证失败，会抛出错误或使用默认形状
```

### 6.3 循环依赖处理
**问题**：操作之间存在形状依赖关系
**解决方案**：
```cpp
// 使用拓扑排序确保依赖顺序
// 或者实现延迟推断机制
if (module::isUnranked(getInput())) {
    // 延迟到运行时推断
    return;
}
```

## 7. 性能优化

### 7.1 缓存机制
```cpp
// 缓存推断结果
static std::map<mlir::Operation*, llvm::SmallVector<int64_t>> shape_cache;

auto it = shape_cache.find(getOperation());
if (it != shape_cache.end()) {
    module::setShapeOrVerify(getOutput(), it->second);
    return;
}
```

### 7.2 批量处理
```cpp
// 批量推断多个操作
void batch_shape_inference(mlir::Block& block) {
    for (auto& op : block) {
        if (auto shape_op = mlir::dyn_cast<ShapeInterface>(op)) {
            shape_op.shape_inference();
        }
    }
}
```

## 8. 监控与维护

### 8.1 性能指标
- 形状推断成功率
- 推断时间开销
- 内存使用情况
- 错误率统计

### 8.2 日志记录
```cpp
void top::ConvOp::shape_inference() {
    LLVM_DEBUG(llvm::dbgs() << "Starting shape inference for ConvOp\n");
    
    // ... 推断逻辑 ...
    
    LLVM_DEBUG(llvm::dbgs() << "Shape inference completed: " 
                            << "input=" << input_shape 
                            << ", output=" << out_shape << "\n");
}
```

### 8.3 持续集成
- 自动化测试
- 性能回归检测
- 代码质量检查
- 文档自动更新

## 9. 总结

形状推断功能的实现是TPU-MLIR项目的重要里程碑，它将：

1. **提升编译质量**：确保生成的代码具有正确的形状信息
2. **增强错误检测**：在编译时发现形状相关问题
3. **优化性能**：基于准确形状信息进行更好的优化
4. **改善用户体验**：减少运行时错误，提高模型部署成功率

通过系统性的实现和测试，我们可以构建一个健壮、高效的形状推断系统，为TPU-MLIR的进一步发展奠定坚实基础。

## 10. 参考资料

- [MLIR Shape Inference Documentation](https://mlir.llvm.org/docs/ShapeInference/)
- [TPU-MLIR Project Repository](https://github.com/sophgo/tpu-mlir)
- [ONNX Shape Inference](https://github.com/onnx/onnx/blob/main/docs/ShapeInference.md)
- [MLIR Operation Definition](https://mlir.llvm.org/docs/DefiningDialects/Operations/) 