# 计算机网络实验报告 LaTeX 模板使用说明

## 文件说明

- `计算机网络实验报告模板.tex` - 主模板文件

## 编译方法

### 使用 XeLaTeX 编译（推荐）

```bash
xelatex 计算机网络实验报告模板.tex
```

### 使用 pdfLaTeX 编译

```bash
pdflatex 计算机网络实验报告模板.tex
```

**注意**：由于使用了 `ctex` 宏包处理中文，推荐使用 XeLaTeX 编译。

## 模板结构

模板包含四个主要部分：

1. **第一页：封面**
   - 课程名称
   - 班级
   - 实验日期
   - 姓名
   - 学号
   - 实验名称
   - 实验目的及要求
   - 实验环境
   - 实验内容

2. **第二页：实验步骤**
   - 用于详细描述实验步骤

3. **第三页：关键问题及分析、总结**
   - 关键问题及分析
   - 总结

4. **第四页：实验成绩评定表**
   - 包含验收和实验报告的评分标准

## 使用方法

### 填写基本信息

在模板中找到对应的位置，将 `\fillblank[6cm]{}` 中的空内容替换为实际信息：

```latex
\textbf{课程名称：} & \fillblank[6cm]{计算机网络} \\
```

### 填写实验步骤

在第二页的 `enumerate` 环境中添加实验步骤：

```latex
\begin{enumerate}[leftmargin=2em, itemsep=1em]
    \item 第一步：配置网络环境
    \item 第二步：编写测试程序
    \item 第三步：运行实验并记录结果
\end{enumerate}
```

### 添加图片

如果需要插入图片，可以使用：

```latex
\usepackage{graphicx}
...
\begin{figure}[htbp]
    \centering
    \includegraphics[width=0.8\textwidth]{图片路径}
    \caption{图片标题}
\end{figure}
```

### 添加代码

可以使用 `listings` 或 `minted` 宏包：

```latex
\usepackage{listings}
\usepackage{xcolor}

\lstset{
    language=Python,
    basicstyle=\ttfamily,
    keywordstyle=\color{blue},
    commentstyle=\color{green},
    numbers=left,
    numberstyle=\tiny,
    frame=single
}

\begin{lstlisting}
# 你的代码
def hello():
    print("Hello, World!")
\end{lstlisting}
```

## 自定义修改

### 调整页面边距

修改 `geometry` 设置：

```latex
\geometry{
    left=2.5cm,    % 左边距
    right=2.5cm,   % 右边距
    top=2.5cm,     % 上边距
    bottom=2.5cm   % 下边距
}
```

### 调整行距

```latex
\singlespacing    % 单倍行距
\onehalfspacing   % 1.5倍行距（默认）
\doublespacing    % 双倍行距
```

### 修改字体大小

在文档类中修改：

```latex
\documentclass[10pt,a4paper]{article}  % 10pt 字体
\documentclass[11pt,a4paper]{article}  % 11pt 字体
\documentclass[12pt,a4paper]{article}  % 12pt 字体（默认）
```

## 常见问题

### Q: 编译时出现中文乱码

A: 确保使用 XeLaTeX 编译，而不是 pdfLaTeX。

### Q: 表格显示不正常

A: 检查是否安装了所有必需的宏包，特别是 `multirow` 和 `array`。

### Q: 如何添加更多页面

A: 使用 `\newpage` 命令开始新页面，然后添加相应的内容。

## 依赖的 LaTeX 宏包

- `ctex` - 中文支持
- `geometry` - 页面设置
- `booktabs` - 表格美化
- `array` - 表格列类型
- `multirow` - 表格多行合并
- `enumitem` - 列表环境定制
- `setspace` - 行距设置

## 示例

完整的使用示例可以参考模板文件中的注释部分。

