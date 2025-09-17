# ARM寻址方式

## 几个概念

- 机器指令：能被微处理器直接识别
- 汇编指令：机器指令的符号化表示形式

## 寻址方式

八种：

1. **寄存器寻址**：直接给出的就是寄存器的值
2. **立即寻址**：操作数就是数据本身
3. **寄存器偏移寻址**：指令中操作数在使用前，先执行移位操作
4. **寄存器间接寻址**：操作数给的是通用寄存器编号，真正数据保存在指定地址地存储单元中
5. **基址寻址**：将基址寄存器的内容与指令中给出的偏移量相加，形成有效地址。
6. **多寄存器寻址**：一次可传送几个寄存器的值。寄存器自己顺序由小到大顺序排列。连续的寄存器可用"-"连接；否则用","分隔书写。

### 多寄存器寻址的地址更新模式

| 模式 | 说明 |
| :--- | :--- |
| **模式 IA** | 每次传送后地址加4 |
| **模式 IB** | 每次传送前地址加4 |
| **模式 DA** | 每次传送后地址减4 |
| **模式 DB** | 每次传送前地址减4 |

> **说明**：
> - **IA (Increment After)**：先传送数据，后增加地址
> - **IB (Increment Before)**：先增加地址，后传送数据
> - **DA (Decrement After)**：先传送数据，后减少地址
> - **DB (Decrement Before)**：先减少地址，后传送数据
> - 这些模式通常与LDM/STM指令配合使用，用于高效访问连续内存块

7. **堆栈寻址**：
    - 堆栈是存储器中一个按特定顺序（先进后厨 or 后进先出）进行存取的区域
    - 使用SP只想一块存储区域，SP指向的存储单元为栈顶。
    - 根据入栈时SP的变化分类
        - 向高地址方向生长，为递增堆栈
        - 向低地址生长，为递减堆栈
    - 根据SP指向内容分类
        - 满堆栈：SP指向最后压入堆栈的有效数据项。
        - 空堆栈：SP指向下一个待压入数据的空位置。

### 四种类型的堆栈（两种分类组合）

| 堆栈类型 | 描述 | 对应指令 |
| :------- | :--- | :------- |
| **满递增 (FA)** | 堆栈向上增长，堆栈指针指向存放有效数据项的最高地址 | LDMFA、STMFA等 |
| **空递增 (EA)** | 堆栈向上增长，堆栈指针指向堆栈上的第一个空位置 | LDMEA、STMEA等 |
| **满递减 (FD)** | 堆栈向下增长，堆栈指针指向存放有效数据项的最低地址 | LDMFD、STMFD等 |
| **空递减 (ED)** | 堆栈向下增长，堆栈指针指向堆栈下的第一个空位置 | LDMED、STMED等 |

> **说明**：
> - **递增/递减**：指堆栈增长方向（向高地址/低地址）
> - **满/空**：指堆栈指针指向的位置（有效数据/空位置）
> - **FD模式**：ARM处理器默认使用的堆栈模式
> - 不同模式对应不同的LDM/STM指令后缀

8. **相对寻址**：基址寻址的变通，PC做基准，操作数做偏移量。

# ARM指令集

ARM920T内核为例（ARMv4）

## 基本格式
$$
<opcode> {<cond>} {S} <Rd> ,<Rn> {,<operand2>}
$$

### 条件码cond

ARM均可以条件执行，Thumb只有B（跳转）指令可以。

| 操作码 | 条件助记符 | 标志 | 含义 |
| :----- | :--------- | :--- | :--- |
| **0000** | EQ | Z=1 | 相等 (即运算结果为零) |
| **0001** | NE | Z=0 | 不相等 (即运算结果不为零) |
| **0010** | CS/HS | C=1 | 无符号数大于或等于 |
| **0011** | CC/LO | C=0 | 无符号数小于 |
| **0100** | MI | N=1 | 负数 |
| **0101** | PL | N=0 | 正数或零 |
| **0110** | VS | V=1 | 溢出 |
| **0111** | VC | V=0 | 没有溢出 |
| **1000** | HI | C=1,Z=0 | 无符号数大于 |
| **1001** | LS | C=0,Z=1 | 无符号数小于或等于 |
| **1010** | GE | N=V | 有符号数大于或等于 |
| **1011** | LT | N!=V | 有符号数小于 |
| **1100** | GT | Z=0,N=V | 有符号数大于 |
| **1101** | LE | Z=1,N!=V | 有符号数小于或等于 |
| **1110** | AL | 任何 | 无条件执行 (指令默认条件) |
| **1111** | NV | 任何 | 从不执行 (不要使用) |

> **说明**：
> - **Z (Zero)**：零标志位，Z=1表示结果为零
> - **C (Carry)**：进位标志位，C=1表示有进位或借位
> - **N (Negative)**：负数标志位，N=1表示结果为负数
> - **V (Overflow)**：溢出标志位，V=1表示有符号数运算溢出
> - 这些条件码用于控制指令的条件执行，提高程序效率

### 标志影响位S

默认不影响CPSR（除了比较指令）

## 指令

### 分支指令

实现程序跳转
- 专门的跳转指令
- 直接向PC写跳转地址值

| 助记符 | 说明 | 操作 | 条件码位置 |
| :----- | :--- | :--- | :--- |
| **B label** | 跳转指令 | PC ← label | B {cond} |
| **BL label** | 带返回的跳转指令 | LR ← PC-4, PC ← label | BL {cond} |
| **BX Rm** | 带状态切换的跳转指令 | PC ← Rm, 切换处理器状态 | BX {cond} |
| **BLX Rm** | 带返回和状态切换的跳转指令 | LR ← PC-4, PC ← Rm, 切换处理器状态 | BLX {cond} |

> **说明**：
> - **PC (Program Counter)**：程序计数器，指向下一条要执行的指令
> - **LR (Link Register)**：链接寄存器，用于保存返回地址
> - **label**：跳转目标地址的标签
> - **Rm**：包含跳转目标地址的寄存器
> - **切换处理器状态**：指在ARM状态和Thumb状态之间切换
> - **{cond}**：可选的条件码，用于条件执行

### 数据处理指令

数据传送指令、算数/逻辑运算指令、比较指令

> 只能对寄存器操作

***ARM数据处理指令----数据传送指令***

| 助记符 | 说明 | 操作 | 条件码位置 |
| :----- | :--- | :--- | :--- |
| **MOV Rd,operand2** | 数据传送 | Rd ← operand2 | MOV{cond}{S} |
| **MVN Rd,operand2** | 数据非传送 | Rd ← (~operand2) | MVN{cond}{S} |

***ARM数据处理指令----比较指令***

| 助记符 | 说明 | 操作 | 条件码位置 |
| :----- | :--- | :--- | :--- |
| **CMP Rn, operand2** | 比较指令 | 标志N、Z、C、V ← Rn - operand2 | CMP{cond} |
| **CMN Rn, operand2** | 负数比较指令 | 标志N、Z、C、V ← Rn + operand2 | CMN{cond} |
| **TST Rn, operand2** | 位测试指令 | 标志N、Z、C、V ← Rn & operand2 | TST{cond} |
| **TEQ Rn, operand2** | 相等测试指令 | 标志N、Z、C、V ← Rn ^ operand2 | TEQ{cond} |

***ARM算术运算指令表***

| 助记符 | 说明 | 操作 | 条件码位置 |
| :----- | :--- | :--- | :--- |
| **ADD Rd, Rn, operand2** | 加法运算指令 | Rd ← Rn + operand2 | ADD{cond}{S} |
| **SUB Rd, Rn, operand2** | 减法运算指令 | Rd ← Rn - operand2 | SUB{cond}{S} |
| **RSB Rd, Rn, operand2** | 逆向减法指令 | Rd ← operand2 - Rn | RSB{cond}{S} |
| **ADC Rd, Rn, operand2** | 带进位加法指令 | Rd ← Rn + operand2 + Carry | ADC{cond}{S} |
| **SBC Rd, Rn, operand2** | 带进位减法指令 | Rd ← Rn - operand2 - (NOT)Carry | SBC{cond}{S} |
| **RSC Rd, Rn, operand2** | 带进位逆向减法指令 | Rd ← operand2 - Rn - (NOT)Carry | RSC{cond}{S} |

***ARM数据处理指令——逻辑运算指令***

| 助记符 | 说明 | 操作 | 条件码位置 |
| :----- | :--- | :--- | :--- |
| **AND Rd, Rn, operand2** | 逻辑与操作指令 | Rd ← Rn & operand2 | AND {cond} {S} |
| **ORR Rd, Rn, operand2** | 逻辑或操作指令 | Rd ← Rn | operand2 | ORR {cond} {S} |
| **EOR Rd, Rn, operand2** | 逻辑异或操作指令 | Rd ← Rn ^ operand2 | EOR {cond} {S} |
| **BIC Rd, Rn, operand2** | 位清除指令 | Rd ← Rn & (~operand2) | BIC {cond} {S} |

> **说明**：
> - **Rd**：目标寄存器
> - **Rn**：第一个操作数寄存器
> - **operand2**：第二个操作数，可以是立即数、寄存器或带移位的寄存器
> - **{cond}**：可选的条件码，用于条件执行
> - **{S}**：可选的S后缀，表示指令执行后更新CPSR中的条件标志位
> - **N (Negative)**：负数标志位
> - **Z (Zero)**：零标志位
> - **C (Carry)**：进位标志位
> - **V (Overflow)**：溢出标志位
> - **~**：按位取反操作
> - **-**：减法操作
> - **+**：加法操作
> - **&**：按位与操作
> - **^**：按位异或操作
> - **|**：按位或操作

### 乘法指令

| 助记符 | 说明 | 操作 | 条件码位置 |
| :----- | :--- | :--- | :--- |
| **MUL Rd,Rm,Rs** | 32位乘法指令 | Rd←Rm*Rs (Rd≠Rm) | MUL{cond}{S} |
| **MLA Rd,Rm,Rs,Rn** | 32位乘加指令 | Rd←Rm*Rs + Rn (Rd≠Rm) | MLA{cond}{S} |
| **UMULL RdLo,RdHi,Rm,Rs** | 64位无符号乘法指令 | (RdLo,RdHi)←Rm*Rs | UMULL{cond}{S} |
| **UMLAL RdLo,RdHi,Rm,Rs** | 64位无符号乘加指令 | (RdLo,RdHi)←Rm*Rs + (RdLo,RdHi) | UMLAL{cond}{S} |
| **SMULL RdLo,RdHi,Rm,Rs** | 64位有符号乘法指令 | (RdLo,RdHi)←Rm*Rs | SMULL{cond}{S} |
| **SMLAL RdLo,RdHi,Rm,Rs** | 64位有符号乘加指令 | (RdLo,RdHi)←Rm*Rs + (RdLo,RdHi) | SMLAL{cond}{S} |

> **说明**：
> - **Rd**：目标寄存器（32位乘法）
> - **RdLo, RdHi**：目标寄存器对（64位乘法），RdLo存储低32位，RdHi存储高32位
> - **Rm, Rs**：源操作数寄存器
> - **Rn**：累加操作数寄存器
> - **{cond}**：可选的条件码，用于条件执行
> - **{S}**：可选的S后缀，表示指令执行后更新CPSR中的条件标志位
> - **约束条件**：Rd≠Rm，确保乘法指令的正确执行
> - **有符号/无符号**：SMULL/SMLAL处理有符号数，UMULL/UMLAL处理无符号数

### 存储器访问指令

包括单寄存器加载/存储指令、多寄存器加载/存储指令、寄存器和存储器交换指令

***ARM存储器访问指令----单寄存器加载指令***

| 助记符 | 说明 | 操作 | 条件码位置 |
| :----- | :--- | :--- | :--- |
| **LDR Rd, addressing** | 加载字数据 | Rd←[addressing] | LDR{cond} |
| **LDRB Rd, addressing** | 加载无符号字节数据 | Rd←[addressing] | LDR{cond}B |
| **LDRT Rd, addressing** | 以用户模式加载字数据 | Rd←[addressing] | LDR{cond}T |
| **LDRBT Rd, addressing** | 以用户模式加载无符号字节数据 | Rd←[addressing] | LDR{cond}BT |
| **LDRH Rd, addressing** | 加载无符号半字数据 | Rd←[addressing] | LDR{cond}H |
| **LDRSB Rd, addressing** | 加载有符号字节数据 | Rd←[addressing] | LDR{cond}SB |
| **LDRSH Rd, addressing** | 加载有符号半字数据 | Rd←[addressing] | LDR{cond}SH |
| **STR Rd, addressing** | 存储字数据 | [addressing] ← Rd | STR{cond} |
| **STRB Rd, addressing** | 存储字节数据 | [addressing] ← Rd | STR{cond}B |
| **STRT Rd, addressing** | 以用户模式存储字数据 | [addressing] ← Rd | STR{cond}T |
| **STRBT Rd, addressing** | 以用户模式存储字节数据 | [addressing] ← Rd | STR{cond}BT |
| **STRH Rd, addressing** | 存储半字数据 | [addressing] ← Rd | STR{cond}H |

> **说明**：
> - **Rd**：源/目标寄存器
> - **addressing**：内存地址，支持多种寻址方式
> - **{cond}**：可选的条件码，用于条件执行
> - **B**：字节操作（8位）
> - **H**：半字操作（16位）
> - **T**：用户模式访问，忽略特权级别检查
> - **S**：有符号扩展
> - **U**：无符号扩展（默认）
> - 加载指令将数据从内存读取到寄存器
> - 存储指令将数据从寄存器写入内存

1. 字和无符号加载/存储指令
    - LDR指令，从内存去一个字/字节，存入寄存器
    - STR指令：与LDR相反
2. 半字和有符号加载/存储指令
    - 可加载有符号半字或字节，可加载/存储无符号半字。

### 多寄存器加载/存储指令

多个寄存器和内存单元兼

- LDM：加载多个连续内存单元内容到多个寄存器
- STM：存储......

$R_n$为基址寄存器：用于存放传送数据的初始地址，不允许为R15（即PC）。

- 后缀！：表示最后的地址要写回$R_n$
- 寄存器列表reglist：可包含多个寄存器或寄存器范围
- 后缀“^”：仅可用于ARM状态。不允许再用户模式或系统模式下使用。

