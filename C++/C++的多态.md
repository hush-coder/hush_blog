# C++的多态

C++ 的多态（Polymorphism）是指同一接口（函数名或操作）在不同类型的对象上表现出不同的行为。它是面向对象编程的核心特性之一，使得代码更灵活、可扩展。C++ 中的多态主要分为静态多态（编译时多态）和动态多态（运行时多态）。

# 1. 静态多态（编译时多态）

在编译阶段就确定具体调用哪个函数，通过**函数重载**、**运算符重载**和**模板**实现。

## 1.1 函数重载

在同一作用域内声明多个同名函数，但参数列表（参数个数、类型或顺序）不同。编译器根据调用时传入的实参类型，在编译期选择最匹配的重载版本。

```cpp
#include <iostream>

void print(int i) {
    std::cout << "整数: " << i << std::endl;
}

void print(double d) {
    std::cout << "浮点数: " << d << std::endl;
}

void print(const char* s) {
    std::cout << "字符串: " << s << std::endl;
}

int main() {
    print(42);        // 调用 print(int)
    print(3.14);      // 调用 print(double)
    print("hello");   // 调用 print(const char*)
}
```

**重载的解析规则**包括精确匹配、提升、标准转换等，由编译器在编译时完成。重载是静态多态的最简单形式，但只能作用于同一作用域内的函数，无法跨类型体系。

## 1.2 运算符重载

本质是函数重载的一种特殊形式，为自定义类型赋予内置运算符的语义，使代码更自然。例如：

```cpp
class Vector {
public:
    Vector operator+(const Vector& other) const { /* ... */ }
};

Vector a, b;
Vector c = a + b;  // 调用 operator+
```

编译器在遇到 `a + b` 时，会查找合适的 `operator+` 重载，这也是编译时决定的。

## 1.3 模板

模板允许将类型作为参数，编译器根据模板实参生成具体代码（实例化），从而实现“一种算法，多种类型”的效果。函数模板和类模板都支持。

### 1.3.1 什么是模版

***1. 函数模板***

```cpp
template<typename T>
T max(T a, T b) {
    return a > b ? a : b;
}

int main() {
    max(3, 5);       // 实例化为 int max(int, int)
    max(3.14, 2.72); // 实例化为 double max(double, double)
    max('a', 'b');   // 实例化为 char max(char, char)
}
```
编译器为每个不同的模板参数类型生成独立的函数版本，这些版本在编译时完全确定，运行时没有任何额外开销。

***2. 类模板***

```cpp
template<typename T>
class Stack {
    std::vector<T> data;
public:
    void push(T value) { data.push_back(value); }
    T pop() { /* ... */ }
};

Stack<int> intStack;   // 生成针对 int 的 Stack 代码
Stack<std::string> strStack; // 生成针对 string 的 Stack 代码
```

类模板使得同一份类定义可以用于多种类型，生成多个独立的类类型。

***3. 模板特化与偏特化***

模板允许为特定类型提供定制实现，进一步增强静态多态的灵活性。

```cpp
// 通用模板
template<typename T>
struct TypeName { static const char* name() { return "unknown"; } };

// 特化
template<>
struct TypeName<int> { static const char* name() { return "int"; } };

template<>
struct TypeName<double> { static const char* name() { return "double"; } };

// 偏特化：针对所有指针类型 T*
template<typename T>
struct TypeName<T*> {
    static const char* name() {
        // 注意：这里需要静态存储，以保证返回的指针有效
        static std::string fullName = std::string(TypeName<T>::name()) + "*";
        return fullName.c_str();
    }
};
```

***4. 静态多态与 CRTP（奇异递归模板模式）***

CRTP 是一种通过模板实现静态多态的惯用法，派生类将自身作为模板参数传递给基类，从而实现基类调用派生类的方法，类似虚函数的效果，但发生在编译期。

```cpp
// ************* 动态多态版本 ************* //
class Animal {
public:
    virtual void speak() const { std::cout << "Animal speaks\n"; }
};

class Dog : public Animal {
public:
    void speak() const override { std::cout << "Dog barks\n"; }
};

class Cat : public Animal {
public:
    void speak() const override { std::cout << "Cat meows\n"; }
};

void makeSound(const Animal& a) {
    a.speak();  // 运行时才知道是 Dog 还是 Cat
}
/*
这里 speak 是虚函数，通过基类指针/引用调用时，程序会根据对象的实际类型（运行时）去查虚函数表，找到正确的函数。这就是动态多态。

缺点：
    1. 每个对象需要存储虚指针（vptr），占用内存。
    2. 调用虚函数需要间接寻址（查表），不能内联（除非编译器激进优化）。
    3. 性能稍低，不适合对性能极敏感的代码。
*/

// ************* 静态多态版本 ************* //
template <typename Animal>
void makeSound(const Animal& a) {
    a.speak();  // 编译时确定 Animal 类型，直接调用对应的方法
}

class Dog {
public:
    void speak() const { std::cout << "Dog barks\n"; }
};

class Cat {
public:
    void speak() const { std::cout << "Cat meows\n"; }
};

int main() {
    Dog d;
    Cat c;
    makeSound(d);  // 编译时生成 makeSound<Dog>，调用 Dog::speak
    makeSound(c);  // 编译时生成 makeSound<Cat>，调用 Cat::speak
}
/*
这里 makeSound 是一个模板，编译器会根据实参类型分别生成两个不同的函数（makeSound<Dog> 和 makeSound<Cat>），里面直接调用对应类型的 speak，没有任何运行时开销。

这就是静态多态：函数调用在编译时已经确定，没有虚函数表，可以内联，性能好。

但是，这种写法有一个问题：Dog 和 Cat 之间没有继承关系，它们只是恰好都有 speak 方法（鸭子类型）。如果我们需要强制它们遵守一个共同的接口（比如都是动物），该怎么办？
*/

// ************* CRTP：奇异递归模版模式 ************* //
template <typename Derived>
class AnimalBase {
public:
    void speak() const {
        // 将 this 转为派生类类型，调用派生类的 speakImpl
        static_cast<const Derived*>(this)->speakImpl();
    }
};

class Dog : public AnimalBase<Dog> {  // 把自己作为模板参数传给基类
public:
    void speakImpl() const {
        std::cout << "Dog barks\n";
    }
};

class Cat : public AnimalBase<Cat> {
public:
    void speakImpl() const {
        std::cout << "Cat meows\n";
    }
};

// 使用静态多态的函数模板
template <typename T>
void makeSound(const AnimalBase<T>& animal) {
    animal.speak();  // 调用的是 AnimalBase<T>::speak，它会调用派生类的 speakImpl
}

int main() {
    Dog d;
    Cat c;
    makeSound(d);  // 输出 Dog barks
    makeSound(c);  // 输出 Cat meows
}
/*
解释
1. 基类 AnimalBase 是一个模板，它接受一个派生类类型 Derived。
2. 基类中定义了一个非虚函数 speak()，它通过 static_cast 把 this 转换成 const Derived*，然后调用 Derived 的 speakImpl()。
3. 派生类 Dog 继承自 AnimalBase<Dog>，必须实现 speakImpl()。
4. 在 makeSound 中，我们接受 AnimalBase<T> 的引用，调用 speak()，实际执行的是对应派生类的 speakImpl()。
*/
```

CRTP 常用于实现代码复用、静态接口多态，比如 Eigen、Boost 等库中大量使用。

### 1.3.2 模版元编程

模板元编程是利用模板在编译期进行计算和代码生成的技术。它把计算从运行时搬到编译期，以模板实例化的方式执行“程序”，结果可以是类型、常量或函数。

***核心思想***

模板元编程基于以下事实：

- 模板可以带非类型参数（如 `int N`）。
- 模板可以递归实例化。
- 模板特化可以充当“条件分支”。

***示例***

```cpp
// ********* 编译期计算阶乘 ********* //
// 通用模板：递归定义
template<unsigned int N>
struct Factorial {
    static constexpr unsigned int value = N * Factorial<N-1>::value;
};

// 特化作为递归终止条件
template<>
struct Factorial<0> {
    static constexpr unsigned int value = 1;
};

int main() {
    int x = Factorial<5>::value;  // 编译时计算为 120
    // 生成的代码中直接是 120，没有运行时计算
}
/*
Factorial<5>::value 在编译期被展开为 5*4*3*2*1，最终结果为常量 120。整个过程由编译器完成，不占用运行时。
*/

// ********* 类型萃取（判断是否为指针） ********* //
template<typename T>
struct IsPointer {
    static constexpr bool value = false;
};

// 偏特化
template<typename T>
struct IsPointer<T*> {
    static constexpr bool value = true;
};

int main() {
    static_assert(IsPointer<int*>::value == true);  // 编译期断言
    static_assert(IsPointer<int>::value == false);
}
```

***模板元编程的用途***

- **类型计算**：如 std::conditional、std::enable_if，根据条件选择类型。
- **编译期优化**：例如展开循环、生成特定数据结构。
- **策略选择**：根据类型特性选择不同实现（如是否支持快速拷贝）。
- **DSL嵌入**：如表达式模板（Eigen、Blitz++）避免临时对象。

***优缺点***

- **优点**：无运行时开销，错误可在编译期捕获，能实现极致优化。
- **缺点**：语法晦涩，编译时间长，错误信息难读，可维护性差。

# 2. 动态多态（运行时多态）

动态多态是面向对象编程的核心特性，它允许通过基类的指针或引用调用派生类的重写函数，并且在**运行时**根据对象的实际类型决定调用哪个函数。C++ 中通过**继承**和**虚函数**实现动态多态。

## 2.1 动态多态的实现机制

当一个类中包含虚函数（包括从基类继承的虚函数）时，编译器会为该类生成一个**虚函数表（vtable）**。vtable 本质上是一个**函数指针数组**，存储了**该类所有虚函数的地址**。同时，每个**对象**会被添加一个隐藏的指针——**虚指针（vptr）**，指向**所属类的 vtable**。

```cpp
class Animal {
public:
    virtual void speak() const { std::cout << "Animal speaks\n"; }
};

class Dog : public Animal {
public:
    void speak() const override { std::cout << "Dog barks\n"; }
};

class Cat : public Animal {
public:
    void speak() const override { std::cout << "Cat meows\n"; }
};
```
- 编译器会为 `Animal` 生成一个 `vtable`，其中包含 `Animal::speak` 的地址。
- 为 `Dog` 生成一个 `vtable`，其中 `speak` 条目指向 `Dog::speak`。
- 为 `Cat` 生成一个 `vtable`，其中 `speak` 条目指向 `Cat::speak`。
- 每个 `Animal` 对象（以及 `Dog`、`Cat` 对象）都包含一个隐藏的 `vptr`，指向对应类的 `vtable`。

内存布局示意（假设在 64 位系统上）：

```
Animal 对象：
[vptr] -> Animal 的 vtable
（其他成员变量）

Dog 对象：
[vptr] -> Dog 的 vtable
（其他成员变量，包括从 Animal 继承的）
```

## 2.2 动态绑定的过程

当通过基类指针或引用调用虚函数时，例如：

```cpp
void makeSound(const Animal& a) {
    a.speak();   // 虚函数调用
}
```

编译器生成的代码大致如下（伪汇编）：

1. 从对象 `a` 中取出 `vptr`（即对象的首地址处）。
2. 通过 `vptr` 找到 `vtable`（`vptr` 指向 `vtable` 的起始地址）。
3. 在 `vtable` 中找到 `speak` 对应的函数指针（通常是 `vtable` 中的第 0 项，取决于声明顺序）。
4. 调用该函数指针指向的函数。

这个过程称为`动态绑定`或`晚绑定`，因为函数地址直到运行时才能确定。

## 2.3 虚函数的开销

***内存开销***

- 每个对象增加一个 `vptr`（通常 8 字节，64 位系统）。对于小型对象（如只包含一个 `int` 的类），这个开销可能很显著（例如 4 字节数据 + 8 字节指针，内存占用扩大 3 倍）。
- 每个类有一个 `vtable`（整个类共享一份），`vtable` 的大小与虚函数数量成正比。

***时间开销***

- **间接调用**：虚函数调用比普通成员函数调用多一次内存访问（读 vptr 和读函数指针），并且通常无法内联。
- **阻止内联**：因为编译器在编译调用点不知道具体调用哪个函数，无法将函数体展开。不过现代编译器支持**去虚拟化（devirtualization）**优化：如果编译器能够通过静态分析确定对象的实际类型（例如通过类型分析或推测），可能会将虚函数调用转换为直接调用甚至内联。
- **分支预测**：间接调用可能干扰 CPU 的分支预测，但现代 CPU 有间接分支预测器，影响相对较小。

***其他开销***

- 虚函数的存在会稍微增加编译单元的大小（vtable 的生成）。
- 虚函数的定义通常不能放在头文件中内联，除非使用关键技巧（如 final 关键字提示编译器）。

## 2.4 虚析构函数

当通过基类指针删除派生类对象时，如果基类析构函数不是虚函数，则只会调用基类的析构函数，派生类的资源无法释放，导致内存泄漏。

```cpp
Animal* p = new Dog();
delete p;   // 如果 Animal 析构函数非虚，只调用 Animal::~Animal()，Dog 的部分未析构
```

因此，**任何可能被继承的类都应该将析构函数声明为虚函数**（除非你明确禁止通过基类指针删除派生类对象）。


## 2.5 构造函数和析构函数中的虚函数行为

在构造函数或析构函数中调用虚函数时，不会发生动态绑定，而是调用当前正在构造或析构的类所定义的版本（即静态类型对应的函数）。这是因为对象的派生类部分尚未构造完成或已经析构，调用派生类的函数是不安全的。

```cpp
class Animal {
public:
    Animal() { speak(); }   // 调用 Animal::speak，而不是 Dog::speak
    virtual void speak() { cout << "Animal\n"; }
};

class Dog : public Animal {
public:
    Dog() { speak(); }      // 此时 Dog 部分已构造，调用 Dog::speak
    void speak() override { cout << "Dog\n"; }
};
```

构造 Dog 对象时输出：

```text
Animal
Dog
```

## 2.6 在MLIR中的应用

### 第1层：接口声明（抽象基类）

```cpp
// 在 MLIR 中，ShapeInference 是这样声明的：
class ShapeInference {
public:
  virtual void inferShapes(InferenceMode mode) = 0;  // ← 纯虚函数
  virtual ~ShapeInference() = default;  // ← 虚析构函数
};
```

**要点：**

- `virtual` 关键字：告诉编译器这是虚函数
- `= 0`：告诉编译器这是纯虚函数（接口）
- 包含虚函数的类被称为 抽象基类（不能直接实例化）

### 第2层：具体操作的实现（派生类）

```cpp
// ConvOp 实现 ShapeInference 接口
class ConvOp : public Operation, public ShapeInference {
public:
  void inferShapes(InferenceMode mode) override {
    // 具体的卷积形状推导逻辑
    auto in_shape = getInputShape();
    auto kernel_shape = getKernelShape();
    auto strides = getStrides();
    auto paddings = getPaddings();
    
    // 计算输出形状：[(H-Kh+2P)/S] x [(W-Kw+2P)/S]
    std::vector<int64_t> out_shape = {
      (in_shape[2] - kernel_shape[0] + 2*paddings[0]) / strides[0] + 1,
      (in_shape[3] - kernel_shape[1] + 2*paddings[1]) / strides[1] + 1
    };
    
    // 设置输出形状
    getResult().setType(RankedTensorType::get(out_shape, ...));
  }
};

// AddOp 实现 ShapeInference 接口  
class AddOp : public Operation, public ShapeInference {
public:
  void inferShapes(InferenceMode mode) override {
    // 具体的加法形状推导逻辑
    auto a_shape = getOperand(0).getType().getShape();
    auto b_shape = getOperand(1).getType().getShape();
    
    // 输出形状：广播规则
    auto out_shape = broadcast_shapes(a_shape, b_shape);
    
    getResult().setType(RankedTensorType::get(out_shape, ...));
  }
};
```

**要点：**

- `override` 关键字：告诉编译器这是重写虚函数
- 每个操作的 `inferShapes()` 实现不同

### 第3层：虚函数表（vtable）机制【核心】

当编译器看到这些类定义时，它会创建虚函数表：

```
┌─────────────────────────────────────────────┐
│ ConvOp 的虚函数表（vtable）                 │
├─────────────────────────────────────────────┤
│ [0] -> ConvOp::inferShapes() 地址: 0x1234 │
│ [1] -> ConvOp::其他虚函数                   │
│ [2] -> ...                                 │
└─────────────────────────────────────────────┘

┌─────────────────────────────────────────────┐
│ AddOp 的虚函数表（vtable）                  │
├─────────────────────────────────────────────┤
│ [0] -> AddOp::inferShapes() 地址: 0x5678  │
│ [1] -> AddOp::其他虚函数                    │
│ [2] -> ...                                 │
└─────────────────────────────────────────────┘
```
> 注意：不同操作有不同的虚函数表！

### 第4层：对象的内存布局【关键】
当创建 ConvOp 对象时：

```cpp
ConvOp *conv = new ConvOp(...);
ShapeInference *shapeOp = conv;  // 基类指针指向派生类对象
```

内存中的布局：

```
┌─────────────────────────────────────────────────┐
│ ConvOp 对象内存布局                              │
├─────────────────────────────────────────────────┤
│ [vptr] -> 指向 ConvOp 的虚函数表                │
│                                                  │
│ 实际数据字段（操作数、结果等）                  │
│                                                  │
└─────────────────────────────────────────────────┘
         ↑                              ↑
         │                              │
         │          ┌──────────────────┴─────────────┐
         │          │ 这是指向虚函数表的指针        │
         │          └───────────────────────────────┘
    shapeOp 指针
```

### 第5层：动态分发过程【最关键！】

```cpp
shapeOp->inferShapes(this->inference_mode);
```

***编译器生成的代码（伪代码）：***

```cpp
// 编译器实际生成的代码
(*shapeOp->vptr[0])(shapeOp, this->inference_mode);
//  ↑          ↑        ↑
// 虚函数表指针  第一个  传递对象
//              虚函数  指针作为this
```

***执行步骤：***

1. **步骤1**：访问 shapeOp 指向的对象
2. **步骤2**：通过对象的 vptr 获取虚函数表
3. **步骤3**：从虚函数表取出第0个函数地址（inferShapes）
4. **步骤4**：跳转到该地址执行（动态绑定！）

***结果：***

- 如果 `shapeOp` 指向 `ConvOp` 对象 → 执行 `ConvOp::inferShapes()`
- 如果 `shapeOp` 指向 `AddOp` 对象 → 执行 `AddOp::inferShapes()`

### 🖼️ 可视化动态分发过程

```
用户代码：
  shapeOp->inferShapes(mode);
       ↑
       ┌───────────────┐
       │ 开始执行      │
       └───────────────┘
              ↓
┌─────────────────────────────────┐
│ 从 shapeOp 指针读取对象的 vptr   │
└─────────────┬───────────────────┘
              ↓
┌─────────────────────────────────┐
│ 通过 vptr 跳转到虚函数表         │
└─────────────┬───────────────────┘
              ↓
┌─────────────────────────────────┐
│ 根据对象的实际类型：             │
│                                 │
│ 如果是 ConvOp:                  │
│   找到 ConvOp 的 vtable[0]      │
│   → ConvOp::inferShapes()       │
│                                 │
│ 如果是 AddOp:                   │
│   找到 AddOp 的 vtable[0]       │
│   → AddOp::inferShapes()        │
└─────────────┬───────────────────┘
              ↓
┌─────────────────────────────────┐
│ 执行具体操作自己的形状推导逻辑  │
└─────────────────────────────────┘
```

### 💡 为什么 Dyn_cast 是前提？

```cpp
if (auto shapeOp = dyn_cast<ShapeInference>(op)) {
    //              ↑
    //              │
    //         动态类型检查
    //              │
    //  如果 op 没有实现 ShapeInference：
    //    dyn_cast 返回 nullptr
    //    不会进入 if 分支
    //    
    //  如果 op 实现了 ShapeInference：
    //    dyn_cast 成功转换
    //    指向 ShapeInference 接口的指针
    //    才能调用 inferShapes()
    //         ↓
  shapeOp->inferShapes(mode);
}
```

Dyn_cast 的作用：

- 确保 `op` 确实实现了 `ShapeInference` 接口
- 返回指向接口的指针（而不是`nullptr`）
- 这样才能安全地调用 `inferShapes()`

## 2.7 dynamic_cast

`dynamic_cast` 是 C++ 中用于**运行时类型识别**的强制转换运算符，主要用于安全地将**基类指针或引用转换为派生类指针或引用**（向下转换），也支持在多重继承中**跨层级转换**（如从某个基类转换到另一个兄弟基类）。它依赖于**运行时类型信息**（RTTI），因此要求类至少有一个虚函数（否则无法使用）。

### 2.7.1 基本用法

```cpp
class Base { virtual void foo() {} };
class Derived : public Base {};

Base* pb = new Derived;
Derived* pd = dynamic_cast<Derived*>(pb);  // 成功，pd 指向 Derived 对象

Base* pbase = new Base;
Derived* pder = dynamic_cast<Derived*>(pbase); // 失败，pder 为 nullptr
```

# 3. 动态 v/s 静态

| 特性 | 静态多态（编译时） | 动态多态（运行时） |
| :--- | :--- | :--- |
| 绑定时机 | 编译期 | 运行期 |
| 实现方式 | 重载、模板 | 虚函数、继承 |
| 速度 | 快（无间接调用） | 略慢（一次间接寻址） |
| 灵活性 | 较低（类型必须编译时确定） | 高（运行时决定类型） |
| 典型应用 | 泛型编程（STL） | 面向对象接口、插件化架构 |

## 3.1 开销对比

### 3.1.1 动态多态的开销（虚函数机制）

1. **间接寻址**：调用虚函数时，编译器无法直接知道调用哪个函数，需要通过对象的虚指针（vptr）找到虚函数表（vtable），再从表中取出函数地址，最后间接调用。这个过程至少需要两次内存访问（读 vptr，读函数指针），并且通常无法内联。
2. **运行时类型解析**：每个虚函数调用都依赖于对象的实际类型，这个类型只有在运行时才能确定，因此编译器无法在编译期进行优化（如内联、常量传播）。
3. **内存占用**：每个对象需要额外存储一个虚指针（通常 8 字节），对于小型对象可能增加显著开销。

### 3.1.2 静态多态（模板/CRTP）的零开销

- **编译时绑定**：模板在实例化时，编译器已经知道具体类型（如 Dog），因此可以直接生成调用 `Dog::speak()` 的代码，就像普通函数调用一样。
- **内联可能**：由于调用目标在编译时确定，编译器可以轻松地将函数体内联到调用点，消除调用开销。
- **无虚表**：不需要虚指针，对象内存更紧凑，也省去了间接寻址的指令。

# 4. 总结

C++ 多态通过静态和动态两种形式，既保留了编译时的高效，又提供了运行时的灵活。理解虚函数表机制和正确使用虚析构函数，是编写健壮可扩展 C++ 程序的基础。在实际开发中，根据场景选择合适的多态方式：需要极致性能且类型固定时用模板；需要运行时多态和接口抽象时用虚函数。