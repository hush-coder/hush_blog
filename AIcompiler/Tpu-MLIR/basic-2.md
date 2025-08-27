***TPU-MLIR-basic篇（2）***

# 摘要

本篇主要介绍 MLIR 框架中 TableGen 的基础语法、Op/Type/Attr 的定义方式、Trait 机制、属性系统、以及 ODS（Operation Definition Specification）与 C++ 的结合实践。内容涵盖 TableGen 的 class/def、trait、属性、code、Dag、methods、DeclareOpInterfaceMethods 等核心概念，并通过丰富实例详细讲解其用法，帮助读者系统理解 MLIR 编译器元编程的设计思想与工程实现。

# 目录

- [摘要](#摘要)
- [目录](#目录)
- [前言](#前言)
- [Op定义方式](#op定义方式)
  - [C++中定义](#c中定义)
  - [Operation Definition Specification(ODS)](#operation-definition-specificationods)
    - [几个问题](#几个问题)
  - [OpBase.td](#opbasetd)
- [TableGen语法](#tablegen语法)
  - [类与定义](#类与定义)
  - [值的定义](#值的定义)
  - [数据类型与表达式](#数据类型与表达式)
    - [Trait](#trait)
    - [code](#code)
    - [Dag](#dag)
- [Trait](#trait-1)
  - [OpTrait](#optrait)
    - [1. ODS（TableGen）方式](#1-odstablegen方式)
      - [1.1 定义 traitbase（通常在 OpBase.td 中已有）](#11-定义-traitbase通常在-opbasetd-中已有)
      - [1.2 定义自定义 trait](#12-定义自定义-trait)
      - [1.3 在 Op 中使用 trait](#13-在-op-中使用-trait)
      - [1.4 添加自定义 verifier（可选）](#14-添加自定义-verifier可选)
    - [2. C++ 方式](#2-c-方式)
      - [2.1 定义 traitbase（MLIR 已有）](#21-定义-traitbasemlir-已有)
      - [2.2 定义自定义 trait](#22-定义自定义-trait)
      - [2.3 在 Op 中使用 trait](#23-在-op-中使用-trait)
    - [3. 解释与总结](#3-解释与总结)
  - [methods 与 DeclareOpInterfaceMethods](#methods-与-declareopinterfacemethods)
    - [1. methods](#1-methods)
      - [示例](#示例)
      - [用法场景](#用法场景)
    - [2. DeclareOpInterfaceMethods](#2-declareopinterfacemethods)
      - [背景](#背景)
      - [作用](#作用)
      - [示例](#示例-1)
  - [3. 总结对比](#3-总结对比)

---

# 前言

本人正在搞AI编译器，这个博客大家可以当作学习笔记

# Op定义方式

## C++中定义

- 对于每个Dialect都要**继承Op**积累并**重写**部分构造函数
- **每个Op都要**编写相应的C++代码
- 冗余
- 可读性差

## Operation Definition Specification(ODS)

- 在**td文件**中编写Op定义
- 利用**TableGen**自动生成相应的C++代码
- Op定义简易直观

---
### 几个问题

***什么是TableGen？***

- TableGen 是 LLVM/MLIR 生态中的一个**元编程工具**。
- 通过声明式语法，描述各种结构、属性、规则，**自动**生成C++代码，减少重复劳动。

***什么是.td文件？***

td文件指的是`TableGen`定义文件,是一种专门用于描述和生成代码结构的声明式配置文件，常用于定义`Dialect`、`Op`、`Type`等元信息。

主要用于：

- 定义`Dialect`（方言/语法扩展）
- 定义`Operation`（算子/操作）
- 定义`Type`（类型）
- 定义`Attribute`（属性）
- 生成相关的C++类、接口、注册代码等

示例：

```tablegen
// MyOps.td
def MyDialect : Dialect {
  let name = "mydialect";
}

def My_AddOp : MyDialect_Op<"add", [NoSideEffect]> {
  let summary = "加法操作";
  let arguments = (ins F32:$lhs, F32:$rhs);
  let results = (outs F32:$result);
}
```

---

## OpBase.td

`OpBase.td` 是 MLIR（和 LLVM）中 `TableGen` 的一个基础定义文件。

其主要公共结构包含：

- **Op**：所有操作（Operation）的基类模板。
- **Attr**：Attribute的基类。
- **Type**：Type的基类。
- **Dialect**：Dialect的基类。

常用的属性和模板：

- **Argument / Result：** 用于描述Op的输入（ins）和输出（outs）。
- **ins / outs**：用于声明Op的输入输出参数列表。
- **AnyType / AnyAttr**：泛型类型/属性匹配模板。
- **Variadic / Optional**：可变参数、可选参数模板。
- **Pred**：用于定义类型/属性的约束条件。
- **StrAttr, I32Attr, F32Attr 等**：常用的属性类型定义（字符串、整型、浮点型等）。
- **TypeAttr, ArrayAttr, DenseElementsAttr 等**：复杂属性类型定义。

约束与辅助结构：

- **TypeConstraint / AttrConstraint**：类型和属性的约束基类。
- **CPred**：用于自定义C++谓词约束。
- **EnumAttr**：枚举类型属性定义模板。
- **DefaultValuedAttr**：带默认值的属性定义模板。
- **OpTrait**：操作特性（如无副作用、可折叠等）的定义模板。
- **OpInterface**：操作接口定义模板。

# TableGen语法

## 类与定义

- **class**：可作为模板或基类去派生子类

```tablegen
class MyOp<string name> {
  let summary = name # " operation";
}
```

- **def**：不可作为模板或基类，可以用class的特化来声明，也可单独使用

```tablegen
def MyAddOp : MyOp<"add"> {
  let arguments = (ins F32:$lhs, F32:$rhs);
  let results = (outs F32:$result);
}
```

> class和def类似于C++中的class和对象

## 值的定义

- let：用于改变def中的值的内容

## 数据类型与表达式

### Trait

**Trait**（特性）是用来描述操作（Op）、类型（Type）等的行为特征的标签。

在 `TableGen` 里，`Trait` 通常是一个 `class`，可以被 `def` 继承，用于为操作添加特定的行为或约束。

```tablegen
def MyOp : MyDialect_Op<"foo", [NoSideEffect, Pure]> {
  // ...
}
```

> 这里 [NoSideEffect, Pure] 就是 Trait list，表示该操作没有副作用且是纯函数。
>> **NoSideEffect**：无副作用
>>
>> **Pure**：纯操作;
>>
>> **SameOperandsAndResultType**：输入输出类型一致
>>
>> **HasCanonicalizer**：有规范化器

### code

code 是 TableGen 的一种特殊类型，用于存储一段原样嵌入的 C++ 代码块。

```tablegen
let verifier = [{
  if ($lhs.getType() != $rhs.getType())
    return emitError("Operands must have the same type");
  return success();
}];
```
> `[{ ... }]` 表示 code 块，内容会被原样插入生成的 C++ 文件中。

### Dag

即是（operator arg0, arg1, arg2）可嵌套的有向无环图。


# Trait

Trait都是traitbase的基类，包含OpTrait、AttrTrait、TypeTrait，这里以OpTrait为例。

## OpTrait

在 MLIR 中，OpTrait 是操作（Op）特性的基类。下面分别用 ODS（TableGen）和 C++ 两种方式，举例说明如何自定义和使用 OpTrait。

### 1. ODS（TableGen）方式

#### 1.1 定义 traitbase（通常在 OpBase.td 中已有）

```tablegen
// OpTrait 是所有操作 trait 的基类
class OpTrait<string traitName> {
  let trait = traitName;
}
```

#### 1.2 定义自定义 trait

```tablegen
// 定义一个自定义 trait，要求第一个输入是整数类型
// 继承自 OpTrait，传入特性名称

def FirstOperandIsInt : OpTrait<"FirstOperandIsInt">;
```

#### 1.3 在 Op 中使用 trait

```tablegen
def MyCustomOp : MyDialect_Op<"custom", [NoSideEffect, SameOperandsAndResultType, FirstOperandIsInt]> {
  let summary = "自定义操作，要求所有输入输出类型一致，且第一个输入为整数";
  let arguments = (ins I32:$a, F32:$b);
  let results = (outs I32:$res);
}
```

- 这里 trait list `[NoSideEffect, SameOperandsAndResultType, FirstOperandIsInt]` 表示该操作无副作用、输入输出类型一致、且第一个输入为整数。

#### 1.4 添加自定义 verifier（可选）

```tablegen
let verifier = [{
  if (!getOperand(0).getType().isa<IntegerType>())
    return emitOpError("第一个输入必须是整数类型");
  return success();
}];
```

- 这样可以在 TableGen 生成的 C++ 代码中自动插入类型检查逻辑。

---

### 2. C++ 方式

#### 2.1 定义 traitbase（MLIR 已有）

```cpp
// OpTrait 是 MLIR 内部的 trait 基类
template <typename ConcreteType>
class OpTrait {};
```

#### 2.2 定义自定义 trait

```cpp
// 自定义 trait，要求第一个输入是整数类型
template <typename ConcreteType>
class FirstOperandIsIntTrait : public OpTrait<ConcreteType> {
public:
  static ::mlir::LogicalResult verifyTrait(::mlir::Operation *op) {
    if (!op->getOperand(0).getType().isa<::mlir::IntegerType>())
      return op->emitOpError("第一个输入必须是整数类型");
    return ::mlir::success();
  }
};
```

#### 2.3 在 Op 中使用 trait

```cpp
class MyCustomOp : public ::mlir::Op<MyCustomOp,
    ::mlir::OpTrait::NoSideEffect,
    ::mlir::OpTrait::SameOperandsAndResultType,
    FirstOperandIsIntTrait> {
  // ... 其他定义 ...
  static ::llvm::StringRef getOperationName() { return "mydialect.custom"; }
  // ... 其他接口 ...
};
```

- 这里 MyCustomOp 继承了多个 trait，包括自定义的 FirstOperandIsIntTrait。

---

### 3. 解释与总结

- **OpTrait** 是 trait 的基类，所有 trait 都继承自它（ODS 里用 `OpTrait<"TraitName">`，C++ 里用 `OpTrait<ConcreteType>`）。
- **自定义 trait** 可以用 TableGen 的 def 语句声明（ODS），也可以用 C++ 模板类实现。
- **使用 trait** 时，在 ODS 里写在 Op 的 trait list 里，在 C++ 里作为模板参数传给 Op。
- **自定义行为**（如类型检查）可以通过 `verifier` 字段（ODS）或 `verifyTrait` 静态方法（C++）实现。

这样，OpTrait 机制就能灵活地为操作添加各种行为和约束，实现高效的元编程和代码生成。

## methods 与 DeclareOpInterfaceMethods<Interface>

在 MLIR TableGen/ODS 中，`methods` 和 `DeclareOpInterfaceMethods<Interface>` 都用于为操作（Op）或接口（Interface）声明 C++ 成员函数，但用途和场景不同。

### 1. methods

`methods` 字段用于为 Op 或 Interface 添加自定义的 C++ 成员函数。这些函数会被自动生成到对应的 C++ 类中，方便在 C++ 代码里直接调用。

#### 示例

```tablegen
let methods = [
  {
    /// 获取操作的输入数量
    unsigned getNumInputs() {
      return this->getNumOperands();
    }
  }
];
```

- 这里 `methods` 是一个列表，每个元素都是一段 C++ 成员函数代码。
- 这些函数会被插入到 TableGen 生成的 Op 类中。

#### 用法场景

- 为 Op/Type/Attr/Interface 添加自定义的工具函数、辅助方法等。
- 让 C++ 端的 API 更加丰富和易用。

---

### 2. DeclareOpInterfaceMethods<Interface>

`DeclareOpInterfaceMethods<Interface>` 是 TableGen 的一个模板，用于**自动为 Op 声明并实现某个接口（Interface）中的所有方法**。

#### 背景
MLIR 支持为 Op 定义“接口”（Interface），接口是一组可以被多个 Op 共享的 API（比如 shape 推断、常量折叠等）。TableGen 里，接口通常用 `OpInterface` 定义。

#### 作用

- `DeclareOpInterfaceMethods<Interface>` 会自动为 Op 类声明并实现该接口的所有方法（包括默认实现和虚函数）。
- 这样你只需在 trait 列表里加上 `DeclareOpInterfaceMethods<MyInterface>`，就能让 Op 支持该接口的所有 API。

#### 示例

假设你有如下接口定义：
```tablegen
def MyShapeInterface : OpInterface<"MyShapeInterface"> {
  let methods = [
    {
      // 形状推断方法
      LogicalResult inferShape(Operation *op);
    }
  ];
}
```

在 Op 定义中使用：

```tablegen
def MyOp : ..., DeclareOpInterfaceMethods<MyShapeInterface> {
  // ...
}
```

- 这样，`MyOp` 就自动拥有了 `MyShapeInterface` 的所有方法声明和实现（如果有默认实现）。

---

## 3. 总结对比

| 名称                           | 作用                                                         | 用法示例                                      |
|--------------------------------|--------------------------------------------------------------|-----------------------------------------------|
| methods                       | 为 Op/Interface 添加自定义 C++ 成员函数                      | let methods = [ { ...C++函数... } ];          |
| DeclareOpInterfaceMethods<IF>  | 自动为 Op 声明并实现某接口的所有方法（批量导入接口 API）      | DeclareOpInterfaceMethods<MyShapeInterface>   |

- `methods` 适合写自定义的、零散的成员函数。
- `DeclareOpInterfaceMethods<Interface>` 适合批量导入接口的所有 API，便于 Op 支持标准化的扩展能力。

这样可以让 Op 的 C++ 端功能更强大、接口更统一，极大提升代码复用性和可维护性。

