***compiler-PKUcompiler flex***

# 目录

# ENBF

```
CompUnit  ::= FuncDef;

FuncDef   ::= FuncType IDENT "(" ")" Block;
FuncType  ::= "int";

Block     ::= "{" Stmt "}";
Stmt      ::= "return" Number ";";
Number    ::= INT_CONST;
```

# flex 与 Bison

## 主要结构

```bison
// 这里写一些选项, 可以控制 Flex/Bison 的某些行为

%{

// 这里写一些全局的代码
// 因为最后要生成 C/C++ 文件, 实现主要逻辑的部分都是用 C/C++ 写的
// 难免会用到头文件, 所以通常头文件和一些全局声明/定义写在这里

%}

// 这里写一些 Flex/Bison 相关的定义
// 对于 Flex, 这里可以定义某个符号对应的正则表达式
// 对于 Bison, 这里可以定义终结符/非终结符的类型

%%

// 这里写 Flex/Bison 的规则描述
// 对于 Flex, 这里写的是 lexer 扫描到某个 token 后做的操作
// 对于 Bison, 这里写的是 parser 遇到某种语法规则后做的操作

%%

// 这里写一些用户自定义的代码
// 比如你希望在生成的 C/C++ 文件里定义一个函数, 做一些辅助工作
// 你同时希望在之前的规则描述里调用你定义的函数
// 那么, 你可以把 C/C++ 的函数定义写在这里, 声明写在文件开头

```

## 如何使用

```bash
# C++ 模式
flex -o 文件名.lex.cpp 文件名.l
bison -d -o 文件名.tab.cpp 文件名.y   # 此时 bison 还会生成 `文件名.tab.hpp`
```
> -d参数会同时生成`文件名.tab.hpp`

# 详解flex

## 全览

```flex
%option noyywrap
%option nounput
%option noinput

%{

#include <cstdlib>
#include <string>

// 因为 Flex 会用到 Bison 中关于 token 的定义
// 所以需要 include Bison 生成的头文件
#include "sysy.tab.hpp"

using namespace std;

%}

/* 空白符和注释 */
WhiteSpace    [ \t\n\r]*
LineComment   "//".*

/* 标识符 */
Identifier    [a-zA-Z_][a-zA-Z0-9_]*

/* 整数字面量 */
Decimal       [1-9][0-9]*
Octal         0[0-7]*
Hexadecimal   0[xX][0-9a-fA-F]+

%%

{WhiteSpace}    { /* 忽略, 不做任何操作 */ }
{LineComment}   { /* 忽略, 不做任何操作 */ }

"int"           { return INT; }
"return"        { return RETURN; }

{Identifier}    { yylval.str_val = new string(yytext); return IDENT; }

{Decimal}       { yylval.int_val = strtol(yytext, nullptr, 0); return INT_CONST; }
{Octal}         { yylval.int_val = strtol(yytext, nullptr, 0); return INT_CONST; }
{Hexadecimal}   { yylval.int_val = strtol(yytext, nullptr, 0); return INT_CONST; }

.               { return yytext[0]; }

%%

```

## 拆解

### 定义模式option

```flex
%option noyywrap
%option nounput
%option noinput
```
> 关闭默认的yywrap以及input/unput辅助函数，简化生成代码

### 插入代码

```flex
%{

#include <cstdlib>
#include <string>

// 因为 Flex 会用到 Bison 中关于 token 的定义
// 所以需要 include Bison 生成的头文件
#include "sysy.tab.hpp"

using namespace std;

%}
```
- 这些会原封不动的放在生成的lex.yy.cc里面
- 需要Inlcude一下bison生成的sysy.tab.hpp

### 模型定义（正则宏）

```flex
/* 空白符和注释 */
WhiteSpace    [ \t\n\r]*
LineComment   "//".*

/* 标识符 */
Identifier    [a-zA-Z_][a-zA-Z0-9_]*

/* 整数字面量 */
Decimal       [1-9][0-9]*
Octal         0[0-7]*
Hexadecimal   0[xX][0-9a-fA-F]+
```
用正则表达式对于token进行模式匹配

### 规则定义

```flex
%%

{WhiteSpace}    { /* 忽略, 不做任何操作 */ }
{LineComment}   { /* 忽略, 不做任何操作 */ }

"int"           { return INT; }
"return"        { return RETURN; }

{Identifier}    { yylval.str_val = new string(yytext); return IDENT; }

{Decimal}       { yylval.int_val = strtol(yytext, nullptr, 0); return INT_CONST; }
{Octal}         { yylval.int_val = strtol(yytext, nullptr, 0); return INT_CONST; }
{Hexadecimal}   { yylval.int_val = strtol(yytext, nullptr, 0); return INT_CONST; }

.               { return yytext[0]; }

%%
```

解释如下：

```flex
%%

// 匹配空白符，直接跳过不返回 token。
{WhiteSpace}    { /* 忽略, 不做任何操作 */ }
// 匹配以 // 开始的单行注释，跳过。
{LineComment}   { /* 忽略, 不做任何操作 */ }

// 关键字，返回 Bison 中的对应token。
"int"           { return INT; }
"return"        { return RETURN; }

// 普通标识符：将词素文本存入语法值 yylval.str_val，返回 IDENT token。
// 这里的yylval代表当前要返回的 token 的语义值容器，词法器匹配到词素后，把对应数据写进 yylval，然后返回一个 token 类型
// str_val 是在 Bison 中定义的 YYSTYPE 联合体/结构体里的一个成员，用来承载标识符的字符串值。你在 Flex 里给 yylval。str_val 赋值时，就是在填充即将返回的 token 的语义值
// 返回时 return IDENT;（或 INT_CONST 等）才是 token 类型，yylval 里的字段是该 token 附带的语义数据。
{Identifier}    { yylval.str_val = new string(yytext); return IDENT; }

{Decimal}       { yylval.int_val = strtol(yytext, nullptr, 0); return INT_CONST; }
{Octal}         { yylval.int_val = strtol(yytext, nullptr, 0); return INT_CONST; }
{Hexadecimal}   { yylval.int_val = strtol(yytext, nullptr, 0); return INT_CONST; }

// 兜底规则：匹配任意单字符，直接返回该字符的 ASCII 码作为 token（便于处理符号如 + - * / 等）。
.               { return yytext[0]; }

%%
```