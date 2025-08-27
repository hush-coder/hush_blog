***AMP-algorithm篇（1）遗传算法GA***

## 目录

- [AI编译器-算法篇（1）遗传算法GA](#ai编译器-算法篇1遗传算法ga)
  - [目录](#目录)
  - [摘要](#摘要)
  - [前言](#前言)
  - [目前问题简述](#目前问题简述)
  - [什么是GA](#什么是ga)
    - [核心流程](#核心流程)
      - [1. 初始化种群](#1-初始化种群)
      - [2. 适应度评估](#2-适应度评估)
      - [3. 选择（Selection）](#3-选择selection)
      - [4. 交叉（Crossover）](#4-交叉crossover)
      - [5. 变异（Mutation）](#5-变异mutation)
      - [6. 终止条件](#6-终止条件)
      - [7. 主流程](#7-主流程)
  - [离散优化问题](#离散优化问题)
    - [定义](#定义)
    - [在混合精度优化中](#在混合精度优化中)
  - [未来的方向](#未来的方向)
    - [GA的变种](#ga的变种)
      - [多目标遗传算法NSGA-II](#多目标遗传算法nsga-ii)
    - [与其他结合](#与其他结合)
      - [模拟退火SA](#模拟退火sa)
      - [禁忌搜索TS](#禁忌搜索ts)
      - [社区分层搜索HiFRTuner](#社区分层搜索hifrtuner)

## 摘要

本文介绍了遗传算法（GA）的基本原理与核心流程，包括初始化种群、适应度评估、选择、交叉、变异等步骤，并结合AI编译器中的混合精度优化问题，阐述了GA在离散优化中的实际应用。文中还展望了多目标遗传算法（如NSGA-II）、模拟退火、禁忌搜索、社区分层搜索等算法的结合与发展方向，为后续优化算法的研究和工程实现提供了参考。

## 前言

本人正在搞AI编译器，这个博客大家可以当作学习笔记

## 目前问题简述

正在做混合精度优化的pass，需要对搜出的变量进行离散优化问题的搜索，所以目前第一个想到的是GA搜索算法。

## 什么是GA

全称**遗传算法**，一种基于自然选择和遗传机制的**元启发式**优化算法。

### 核心流程

#### 1. 初始化种群

随机生成一组候选解（称为 **“个体”** 或 **“染色体”**），通常用二进制串、实数向量或其他编码形式表示。

```python
def create_initial_population(self) -> List[Dict[str, Any]]:
        """创建初始种群"""
        population = []
        
        # 添加所有初始配置作为个体
        for i, config in enumerate(self.initial_configs):
            population.append(copy.deepcopy(config))
            print(f"添加初始配置 {i+1} 到种群")
        
        # 生成随机个体填充剩余位置
        remaining_size = self.population_size - len(self.initial_configs)
        for i in range(remaining_size):
            individual = self.create_random_individual()
            population.append(individual)
            print(f"添加随机个体 {i+1} 到种群")
        
        print(f"初始种群大小: {len(population)}")
        return population
```

#### 2. 适应度评估

计算每个个体的**适应度值**（Fitness），反映其解的质量（如目标函数值）。

```python
def evaluate_fitness(self, config: Dict[str, Any], individual_id: int) -> float:
        """
        评估适应度（以final_marks为标准）
        """
        # 创建个体输出目录
        individual_dir = os.path.join(self.output_base, f"individual_{individual_id}")
        os.makedirs(individual_dir, exist_ok=True)
        
        # 创建arm64输出目录
        arm64_output_dir = os.path.join(individual_dir, "arm64_output")
        os.makedirs(arm64_output_dir, exist_ok=True)
        
        # 确保在工作目录下创建必要的文件
        exclude_file = os.path.join(individual_dir, "exclude.txt")
        include_file = os.path.join(individual_dir, "include.txt")
        if not os.path.exists(exclude_file):
            open(exclude_file, 'w').close()
        if not os.path.exists(include_file):
            open(include_file, 'w').close()
        
        # 此处省略一百行代码......

                    # 打印基准值
                    print(f"=== 基准配置性能 ===")
                    print(f"基准最小性能: {min_T0:.4f} Gflops")
                    print(f"基准平均性能: {mean_T0:.4f} Gflops")
                    print(f"基准最大性能: {max_T0:.4f} Gflops")
                    print(f"基准配置输出:")
                    print(baseline_output)
                    print(f"=========================")

                else:
                    min_T0, mean_T0, max_T0 = self._baseline_T0
                # 计算flops_marks和final_marks
                flops_marks = 100 * (0.6 * max_flops / max_T0 + 0.3 * mean_flops / mean_T0 + 0.1 * min_flops / min_T0) / 2
                final_marks = pass_rate * 100 * 0.4 + flops_marks * 0.6
                print(f"Individual {individual_id} final_marks: {final_marks:.4f}")
                return -final_marks
            else:
                return float('inf')
                
        except subprocess.TimeoutExpired:
            print(f"Timeout for individual {individual_id}")
            return float('inf')
        except Exception as e:
            print(f"Error evaluating individual {individual_id}: {e}")
            return float('inf')
    
```

> 此处代码是为了调用pass实现降精，然后运行二进制测试文件来进行适应度评估，这个需要根据实际问题来。
>
> 具体代码就不贴了，不然有人说我水字数hhhh（bushi）

#### 3. 选择（Selection）

根据适应度选择优秀个体进入下一代（常用方法：轮盘赌选择、锦标赛选择）。

```python
def tournament_selection(self, population: List[Dict[str, Any]], 
                           fitness_values: List[float], tournament_size: int = 3) -> Dict[str, Any]:
        """锦标赛选择"""
        tournament_indices = random.sample(range(len(population)), tournament_size)
        tournament_fitness = [fitness_values[i] for i in tournament_indices]
        
        # 选择适应度最好的个体（适应度值越小越好，因为返回的是负的Gflops值）
        winner_index = tournament_indices[tournament_fitness.index(min(tournament_fitness))]
        return copy.deepcopy(population[winner_index])
```

> 这里是**锦标赛选择**
>
> 所谓**轮盘赌**其实就是根据适应度算出其被抽到的概率。

#### 4. 交叉（Crossover）

模拟基因重组，随机配对父代个体并**交换部分编码**（如单点交叉、多点交叉）。

```python
def crossover(self, parent1: Dict[str, Any], parent2: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """交叉操作"""
        if random.random() > self.crossover_rate:
            return copy.deepcopy(parent1), copy.deepcopy(parent2)
        
        child1 = copy.deepcopy(parent1)
        child2 = copy.deepcopy(parent2)
        
        # 对每个变量进行交叉
        for i, var in enumerate(child1.get("localVar", [])):
            if random.random() < 0.5:
                # 交换类型
                child1["localVar"][i]["type"] = parent2["localVar"][i]["type"]
                child2["localVar"][i]["type"] = parent1["localVar"][i]["type"]
        
        return child1, child2
```

#### 5. 变异（Mutation）

以低概率随机改变个体的部分编码（如翻转二进制位），增加种群多样性。

```python
def mutate(self, individual: Dict[str, Any]) -> Dict[str, Any]:
        """变异操作"""
        if random.random() > self.mutation_rate:
            return individual
        
        mutated = copy.deepcopy(individual)
        
        # 随机选择一些变量进行变异
        for var in mutated.get("localVar", []):
            if random.random() < 0.1:  # 10%概率变异每个变量
                var["type"] = random.choice(self.precision_types)
        
        return mutated
```

#### 6. 终止条件

重复步骤2-5，直到满足终止条件（如达到最大迭代次数、适应度收敛等）。

#### 7. 主流程

以下是算法的主流程：

```python
def evolve_population(self, population: List[Dict[str, Any]], 
                         fitness_values: List[float]) -> List[Dict[str, Any]]:
        """进化种群"""
        new_population = []
        
        # 精英保留：保留最好的20%个体
        elite_size = max(1, self.population_size // 5)
        elite_indices = sorted(range(len(fitness_values)), key=lambda i: fitness_values[i])[:elite_size]
        
        for idx in elite_indices:
            new_population.append(copy.deepcopy(population[idx]))
        
        # 生成新个体
        while len(new_population) < self.population_size:
            # 选择父代
            parent1 = self.tournament_selection(population, fitness_values)
            parent2 = self.tournament_selection(population, fitness_values)
            
            # 交叉
            child1, child2 = self.crossover(parent1, parent2)
            
            # 变异
            child1 = self.mutate(child1)
            child2 = self.mutate(child2)
            
            new_population.extend([child1, child2])
        
        # 确保种群大小正确
        return new_population[:self.population_size]
```

## 离散优化问题

### 定义

指在**离散变量**或**有限解空间**中寻找最优解的一类数学优化问题。与连续优化不同，其解集通常是可数的（如整数、集合、排列等），且可能涉及复杂的约束条件。

### 在混合精度优化中

在混合精度优化中，对于优化pass在原测试用例中搜索到的精度配置文件，例如：

```json
{
    "localVar": [
        {
            "function": "gmres",
            "name": "tol",
            "type": "double"
        },
        {
            "function": "gmres",
            "name": "norm_b362",
            "type": "double"
        },
        {
            "function": "gmres",
            "name": "norm_r",
            "type": "double"
        }
    ]
}
```

进行配置的搜索，可以将每一个配置进行抽象，然后投入搜索，找出最佳的配置。

## 未来的方向

### GA的变种

#### 多目标遗传算法NSGA-II

- **主要目标：** 同时优化两个或多个互相冲突的目标，找出一组“互不支配”的最优解（帕累托前沿），这些解有的精度高但性能一般，有的性能高但精度一般，用户可以根据实际需求选择。
- **主要流程：**
  1. **初始化种群：** 与传统GA类似
  2. **评估每个个体的各个目标：** 比如我们这里target1是提升性能，target2是精度损失（这里的两个target必须是相互冲突的）
  3. **非支配排序：** 将解分层：第一层所有非支配解（即“帕累托最优解”）；第二层：去除第一层后，剩下的解中的非支配解；以此类推。
  4. **计算拥挤度距离：** 优先保留那些分散的解
  5. **选择、交叉、变异：** 综合考量非支配层级以及拥挤度，来选择评分较高的优秀解做为父母，产生新一代种群。
  6. **重复进化：** 父代和子代合并后，重新进行非支配排序和拥挤度计算，选出最优的N个解作为新一代，然后重复。
  7. **最终：** 最终得到一组互补支配的解，即帕累托前沿。
- **优点：**
  1. 自动给出多种权衡解，无需与设精度与性能的权重。
  2. 适合多目标明显冲突的情况。


### 与其他结合

#### 模拟退火SA

- 模仿物理退火过程，允许以一定概率接受较差解，逐步降低“温度”。
- **优点：**
  1. 实现简单
  2. 跳出局部最优能力强。
- **缺点：**
   1. 收敛速度较慢
   2. 比较吃初始解的质量
- **与GA的结合：** 可以用GA的结果做初始解，在局部调优。

#### 禁忌搜索TS

- 记录最近访问过的解（禁忌表），也就是记入一个队列，当前解的所有邻域解中，选出最优且不在禁忌表的解
- **优点：**
  1. 能有效避免局部最优
  2. 收敛速度快
  3. 易集成到其他算法中
- **与GA的结合：**
  1. 可以在GA的迭代过程中建立一个禁忌表，可以有效加快迭代速度和跳出局部搜索的能力
  2. 与SA类似，可以微调GA中的精英个体，相较于SA，TS的优点是收敛速度更快，缺点是跳出局部最优能力较弱。

#### 社区分层搜索HiFRTuner

- 构建变量间的加权依赖图（变量的影响程度），按照权重分为若干社区，从大到小依次分层搜索
- **优点：**
  1. 有效降低搜索空间
  2. 并且考虑到了变量之间的依赖关系，能自动发现“影响彼此较大”的变量组，统一调整这些变量的精度，既能保证数值稳定，也能更高效地提升性能。
- **与GA的结合：** 可以分层GA，减少染色体长度以及搜索空间，具体性能是否提升需要实操。