***compiler-PKUcompiler AST类***

# 目录

# AST设计思路

## 设计原则

1. **继承+多态**：所有AST节点继承自`BaseAST`基类
2. **智能指针**：使用`std::unique_ptr`管理子节点，自动释放内存
3. **统一接口**：通过`Dump()`方法实现AST遍历和输出

## 核心类设计

### 1. 基类 BaseAST

```cpp
// include/ast/BaseAST.hpp
#pragma once
#include <iostream>
#include <memory>

class BaseAST{
    public:
        virtual ~BaseAST() = default;
        virtual void Dump() const = 0;
};
```

- 纯虚析构函数确保派生类正确析构
- 纯虚函数`Dump()`强制所有派生类实现

### 2. 编译单元节点 CompUnitAST

```cpp
// include/ast/CompUnitAST.hpp
#pragma once
#include "BaseAST.hpp"
#include "FuncDefAST.hpp"

class CompUnitAST: public BaseAST{
    public:
        std::unique_ptr<FuncDefAST> func_def;
        void Dump() const override;
};
```

```cpp
// src/ast/CompUnitAST.cpp
#include "ast/CompUnitAST.hpp"

void CompUnitAST::Dump() const {
    std::cout << "CompUnitAST {";
    func_def -> Dump();
    std::cout << " }";
}
```

- AST根节点，代表整个编译单元
- 包含一个函数定义子节点

### 3. 函数定义节点 FuncDefAST

```cpp
// include/ast/FuncDefAST.hpp
#pragma once
#include "BaseAST.hpp"
#include "FuncTypeAST.hpp"
#include "BlockAST.hpp"

class FuncDefAST: public BaseAST{
    public:
        std::unique_ptr<FuncTypeAST> func_type;
        std::string ident;
        std::unique_ptr<BlockAST> block;
        void Dump() const override;
};
```

```cpp
// src/ast/FuncDefAST.cpp
#include "ast/FuncDefAST.hpp"

void FuncDefAST::Dump() const {
    std::cout << "FuncDefAST {";
    func_type -> Dump();
    std::cout << " " << ident << " ";
    block -> Dump();
    std::cout << " }";
}
```

- 包含函数类型、函数名和函数体三部分

### 4. 函数类型节点 FuncTypeAST

```cpp
// include/ast/FuncTypeAST.hpp
#pragma once
#include "BaseAST.hpp"

class FuncTypeAST: public BaseAST{
    public:
        std::string type;
        void Dump() const override;
};
```

```cpp
// src/ast/FuncTypeAST.cpp
#include "ast/FuncTypeAST.hpp"

void FuncTypeAST::Dump() const {
    std::cout << "FuncTypeAST {";
    std::cout << type;
    std::cout << " }";
}
```

- 叶子节点，存储函数返回类型（如"int"）

### 5. 语句块节点 BlockAST

```cpp
// include/ast/BlockAST.hpp
#pragma once
#include "BaseAST.hpp"
#include "StmtAST.hpp"

class BlockAST: public BaseAST{
    public:
        std::unique_ptr<StmtAST> stmt;
        void Dump() const override;
};
```

```cpp
// src/ast/BlockAST.cpp
#include "ast/BlockAST.hpp"

void BlockAST::Dump() const {
    std::cout << "BlockAST {";
    stmt -> Dump();
    std::cout << " }";
}
```

- 表示语句块，包含一个语句

### 6. 语句节点 StmtAST

```cpp
// include/ast/StmtAST.hpp
#pragma once
#include "BaseAST.hpp"

class StmtAST: public BaseAST{
    public:
        int number;
        void Dump() const override;
};
```

```cpp
// src/ast/StmtAST.cpp
#include "ast/StmtAST.hpp"

void StmtAST::Dump() const {
    std::cout << "StmtAST {";
    std::cout << number;
    std::cout << " }";
}
```

- 叶子节点，存储语句值（当前为整数）

## AST层次结构

```
CompUnitAST (根节点)
└── FuncDefAST (函数定义)
    ├── FuncTypeAST (函数类型) - 叶子节点: "int"
    ├── ident (函数名) - 字符串值
    └── BlockAST (函数体)
        └── StmtAST (语句) - 叶子节点: 整数
```

## 统一头文件

定义了一个AST.hpp聚合头

```cpp
// include/ast/AST.hpp
#pragma once
#include "BaseAST.hpp"
#include "CompUnitAST.hpp"
#include "FuncDefAST.hpp"
#include "FuncTypeAST.hpp"
#include "BlockAST.hpp"
#include "StmtAST.hpp"
```

## 设计特点

- **多态性**：通过虚函数实现统一接口
- **内存安全**：`unique_ptr`自动管理内存
- **可扩展性**：新增节点只需继承`BaseAST`并实现`Dump()`
- **递归遍历**：`Dump()`方法递归调用子节点，实现树形输出

## 扩展flex/bison

由于之前是简单输出一个string，现在需要进行替换为ast：

```bison
void yyerror(std::unique_ptr<BaseAST> &ast, const char *s);

%parse-param { std::unique_ptr<BaseAST> &ast }

void yyerror(unique_ptr<BaseAST> &ast, const char *s) {
    cerr << "error: " << s << endl;
}
```

这里定义的unique_ptr需要改成BaseAST类型

```bison
%union {
    std::string *str_val;
    int int_val;
    BaseAST *ast_val;
}
```

语义结构体新增一个ast类型

```bison
%token INT RETURN
%token <str_val> IDENT
%token <int_val> INT_CONST

%type <ast_val> FuncDef FuncType Block Stmt 
%type <int_val>Number
```

相应的非终结符和终结符也要修改

```bison
CompUnit
  : FuncDef {
    //这里make_unique是为了管理整体的ast指针
    auto comp_unit = make_unique<CompUnitAST>();
    comp_unit -> func_def = unique_ptr<FuncDefAST>(static_cast<FuncDefAST*>($1));
    ast = move(comp_unit);
  }
  ;

FuncDef
  : FuncType IDENT '(' ')' Block {
    //这里只能new裸指针，因为上面union里面定义的Bison的语义值就是裸指针
    //但是这里的裸指针会交给智能指针comp_unit统一管理
    auto ast = new FuncDefAST();
    ast -> func_type = unique_ptr<FuncTypeAST>(static_cast<FuncTypeAST*>($1));
    ast -> ident = *unique_ptr<string>($2);
    ast -> block = unique_ptr<BlockAST>(static_cast<BlockAST*>($5));
    $$ = ast;
  }
  ;

FuncType
  : INT {
    auto ast = new FuncTypeAST();
    ast -> type = "int";
    $$ = ast;
  }

Block
  : '{' Stmt '}' {
    auto ast = new BlockAST();
    ast -> stmt = unique_ptr<StmtAST>(static_cast<StmtAST*>($2));
    $$ = ast;
  }
  ;

Stmt
  : RETURN Number ';' {
    auto ast = new StmtAST();
    ast -> number = $2;
    $$ = ast;
  }
  ;

Number
  : INT_CONST {
    $$ = $1;
  }
  ;

%%
```

同时在产生式规则这里，也不能只是简单的new string了。

# 编译输出

至此，加上makefile就可以初步输出了:

```makefile
CXX		 := g++
CXXFLAGS := -std=c++17 -Wall -Wextra -g -Iinclude -Ibuild
FLEX	 := flex
BISON	 := bison

SRC_DIR	  := src
BUILD_DIR := build
BIN		  := ./HHSH

BISON_Y	  := $(SRC_DIR)/sysy.y
FLEX_L	  := $(SRC_DIR)/sysy.l

BISON_CPP := $(SRC_DIR)/sysy.tab.cpp
BISON_HPP := $(SRC_DIR)/sysy.tab.hpp
FLEX_CPP  := $(SRC_DIR)/sysy.yy.cpp

SRCS := \
  src/main.cpp \
  $(BISON_CPP) \
  $(FLEX_CPP) \
  src/ast/StmtAST.cpp \
  src/ast/BlockAST.cpp \
  src/ast/FuncTypeAST.cpp \
  src/ast/FuncDefAST.cpp \
  src/ast/CompUnitAST.cpp

OBJS := $(SRCS:src/%.cpp=$(BUILD_DIR)/%.o)

.PHONY: all clean dirs

all: dirs $(BIN)

dirs:
	mkdir -p $(BUILD_DIR)

$(BIN): $(OBJS)
	$(CXX) $(CXXFLAGS) $^ -o $@

$(BISON_CPP) $(BISON_HPP): $(BISON_Y) | dirs
	$(BISON) -d -o $(BISON_CPP) $<

$(FLEX_CPP): $(FLEX_L) | dirs
	$(FLEX) -o $(FLEX_CPP) $<

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp | dirs
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -rf $(BUILD_DIR)
```

输出如下

```bash
root@e2a13c77841a:~/PKU_compiler# ./HHSH -S ./test/init.cpp a a

CompUnitAST {FuncDefAST {FuncTypeAST {int } main BlockAST {StmtAST {0 } } } }
```