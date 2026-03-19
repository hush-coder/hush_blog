# C++的多态

C++ 的多态（Polymorphism）是指同一接口（函数名或操作）在不同类型的对象上表现出不同的行为。它是面向对象编程的核心特性之一，使得代码更灵活、可扩展。C++ 中的多态主要分为静态多态（编译时多态）和动态多态（运行时多态）。

## 1. 静态多态（编译时多态）

在编译阶段就确定具体调用哪个函数，通过**函数重载**、**运算符重载**和**模板**实现。

### 1.1 函数重载

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

### 1.2 运算符重载

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

### 1.3 模板

模板允许将类型作为参数，编译器根据模板实参生成具体代码（实例化），从而实现“一种算法，多种类型”的效果。函数模板和类模板都支持。

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
template<typename Derived>
class Base {
public:
    void interface() {
        static_cast<Derived*>(this)->implementation();
    }
};

class Derived : public Base<Derived> {
public:
    void implementation() { /* 具体实现 */ }
};

int main() {
    Derived d;
    d.interface(); // 调用 Derived::implementation
}
```

CRTP 常用于实现代码复用、静态接口多态，比如 Eigen、Boost 等库中大量使用。


## 2. 动态多态（运行时多态）

在程序运行时根据对象的实际类型决定调用哪个函数，通过继承和虚函数实现。

***核心机制***

- **虚函数**：基类中使用 `virtual` 关键字声明的成员函数，允许派生类重写（`override`）。
- **虚函数表（vtable）**：每个包含虚函数的类都有一个虚函数表，表中存放该类的虚函数地址。
- **虚指针（vptr）**：每个对象内部有一个隐藏指针（`vptr`），指向所属类的虚函数表。

当通过基类指针或引用调用虚函数时，程序会**动态绑定**：从对象的 `vptr` 找到 `vtable`，再从 `vtable` 中取出正确的函数地址执行。

***示例***

```cpp
class Animal {
public:
    virtual void speak() const { cout << "Animal speaks\n"; }
    virtual ~Animal() {}  // 虚析构函数，确保正确释放派生类对象
};

class Dog : public Animal {
public:
    void speak() const override { cout << "Dog barks\n"; }
};

class Cat : public Animal {
public:
    void speak() const override { cout << "Cat meows\n"; }
};

void makeSound(const Animal& a) {
    a.speak();  // 运行时根据实际对象类型调用
}

int main() {
    Dog d;
    Cat c;
    makeSound(d);  // 输出 "Dog barks"
    makeSound(c);  // 输出 "Cat meows"
}
```

***条件***

1. 存在继承关系。
2. 基类中声明虚函数，派生类中重写（override）。
3. 通过基类的指针或引用调用虚函数。

***注意事项***

- **虚析构函数**：如果基类析构函数不是虚函数，删除基类指针指向的派生类对象时，只会调用基类析构函数，导致派生类资源泄漏。
- **纯虚函数与抽象类**：包含纯虚函数（如 virtual void f() = 0;）的类称为抽象类，不能实例化，用于定义接口。
- **性能开销**：虚函数调用比普通函数多一次间接寻址（通过 vptr 和 vtable），且通常无法内联。但现代编译器能进行一定优化。
- **覆盖（override）与隐藏（hide）**：派生类中如果函数名与基类相同但参数不同，会隐藏基类同名函数，而非覆盖。C++11 引入 override 关键字显式标明意图，避免意外隐藏。

## 3. 多态的应用与意义

- **解耦与扩展**：通过基类接口编程，新增派生类无需修改原有代码（开闭原则）。
- **设计模式实现**：如工厂模式、策略模式、观察者模式等大量依赖多态。
- **框架设计**：如 GUI 事件处理、游戏引擎中角色行为定义。

## 4. 总结

| 特性 | 静态多态（编译时） | 动态多态（运行时） |
| :--- | :--- | :--- |
| 绑定时机 | 编译期 | 运行期 |
| 实现方式 | 重载、模板 | 虚函数、继承 |
| 速度 | 快（无间接调用） | 略慢（一次间接寻址） |
| 灵活性 | 较低（类型必须编译时确定） | 高（运行时决定类型） |
| 典型应用 | 泛型编程（STL） | 面向对象接口、插件化架构 |

C++ 多态通过静态和动态两种形式，既保留了编译时的高效，又提供了运行时的灵活。理解虚函数表机制和正确使用虚析构函数，是编写健壮可扩展 C++ 程序的基础。在实际开发中，根据场景选择合适的多态方式：需要极致性能且类型固定时用模板；需要运行时多态和接口抽象时用虚函数。