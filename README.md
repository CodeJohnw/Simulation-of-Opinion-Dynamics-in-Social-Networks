# Simulation-of-Opinion-Dynamics-in-Social-Networks


# 1.Ising 模型

这个项目实现了 Ising 模型的模拟，用于模拟一个人口群体对某个议题持有赞成或反对的观点。每个人被赋予一个值，+1 表示赞成，-1 表示反对。在这个模型中的关键假设是，人们希望与他们的邻居持有相似的观点，如果周围的人都持有不同的观点，他们更有可能改变自己的观点。我们可以通过计算一个人与其周围邻居的一致性来捕捉这个原则。在这个模型中，我们将使用一个 numpy 数组来初始化人口，假设每个人的邻居是网格中直接相邻的人。如果一个人与邻居持有相同的观点，则乘积将为正数。如果一个人持有不同的观点，则乘积的符号将不同，因此乘积将为负数。根据乘积的结果，我们决定是否让这个人改变他的观点。

## 模型扩展

这个简单的模型可以通过两种方式进行扩展。首先，我们可以引入一个概率来表示即使一致性为正数时，有时候也会改变观点的情况。在这个模型中，我们选择接受减少一致性的翻转，概率为 p。其次，我们可以引入外部对观点的影响效应，例如媒体中的主流观点。我们可以通过在计算一致性时，加上一个外部影响的值 H 来实现。

## 测试功能

我们已经提供了一个测试函数 `test_ising`，用于确保代码的正确性。通过运行带有标志 `-test_ising` 的代码，可以验证整体模型的正确性。

## 使用方法

你可以通过命令行来运行这个模型，并根据需要设置不同的参数：

```bash
$ python3 Ising_Model.py -ising_model
# 使用默认参数运行 Ising 模型
$ python3 Ising_Model.py -ising_model -external -0.1
# 使用默认温度和外部影响运行 Ising 模型
$ python3 Ising_Model.py -ising_model -alpha 10
# 使用没有外部影响但有温度的 Ising 模型
$ python3 Ising_Model.py -test_ising
# 运行与模型相关的测试函数
```

## 注意事项

这个模型是用 Python 编写的，使用了 numpy 和 matplotlib 库。请确保你的环境中安装了这些库。


# 2.Defuant模型

## 概述

Defuant模型模拟了个体意见在连续范围[0, 1]内的演化。该模型基于一个原理：人们希望与邻居的意见达成一致，但这种一致性仅限于意见差在某个阈值范围内的邻居。换句话说，人们不愿意听取与自己意见相差过大的人的意见。

我们通过不断选择随机个体及其随机邻居来模拟这种情况。如果所选两人的意见差大于阈值，则不做任何处理。如果意见差小于阈值，则更新双方的意见，使其向平均意见靠拢。数学上可以描述为：
```
if |xi(t) - xj(t)| < T:
    xi(t+1) = xi(t) + beta * (xj(t) - xi(t))
    xj(t+1) = xj(t) + beta * (xi(t) - xj(t))
```
其中，`T`是交互阈值，`beta`是耦合参数。`beta`值大时，每次更新会使双方意见大幅向平均值靠拢；`beta`值小时，每次更新仅会有小幅变化。

你的任务是在一维网格上实现这个模型，这意味着每个人只有两个可能的邻居：左边一个和右边一个。你需要包含一些绘图功能来展示模型的解决方案，并编写一些测试函数来确保代码的正确性。

## 运行与测试

你需要包含一个`test_defuant`函数来检查模型是否正确更新意见。确保涵盖模型的基本行为和任何边界情况。运行你的代码并带上`-test_defuant`标志时，我应该能够调用你的测试函数。

## 模型正确性的判断标准

如果模型正确运行，我应该看到最初随机分布的意见开始分离成不同的集群。增加`beta`值会加速集群的形成，而减小阈值会增加集群的数量。当阈值约为0.5时，模型应演化为单一集群（即所有人趋于一致）。随着阈值的减小，我应该看到更多的集群出现。

## 预期效果

当我运行你的代码并带上`-defuant`标志时，代码应该使用默认参数解决模型并生成类似于下图的图形：

### 运行示例

```bash
$ python3 Defuant_Model.py -defuant
```
这将使用默认参数运行Defuant模型。

```bash
$ python3 Defuant_Model.py -defuant -beta 0.1
```
这将运行Defuant模型，使用默认阈值并设置`beta`为0.1。

```bash
$ python3 Defuant_Model.py -defuant -threshold 0.3
```
这将运行Defuant模型，使用阈值0.3。

```bash
$ python3 Defuant_Model.py -test_defuant
```
这将运行你编写的测试函数。

## 代码示例

以下是主要的代码片段：

```python
import random
import matplotlib.pyplot as plt
import argparse

# 定义最大个体数和模拟时间步长
MAX_PERSON = 100
MAX_TIME = 100

def defuant(beta, threshold):
    """
    运行Defuant模型模拟个体意见的演化。
    
    参数:
    - beta: float, 个体间意见调整的影响系数（耦合参数）。
    - threshold: float, 意见差的交互阈值。
    """
    # 初始化意见数组，每个个体的意见是0到1之间的随机数
    opinion = [random.uniform(0, 1) for _ in range(MAX_PERSON)]
    
    # 创建图形和子图
    fig, axe = plt.subplots(1, 2)
    # 设置图形标题
    fig.suptitle(f'Coupling: {beta}, Threshold: {threshold}')

    # 模拟过程，时间步循环
    for i in range(MAX_TIME):
        # 对每个个体进行操作
        for j in range(MAX_PERSON):
            # 随机选择一个个体A
            A = random.randint(0, MAX_PERSON - 1)
            # 选择个体A的邻居B，处理边界条件
            B = random.choice([A - 1, A + 1]) if 0 < A < MAX_PERSON - 1 else (
                random.choice([1, MAX_PERSON - 1]) if A == 0 else random.choice([0, MAX_PERSON - 2]))
            # 如果A和B的意见差小于阈值，则调整他们的意见
            if abs(opinion[A] - opinion[B]) <= threshold:
                oA, oB = opinion[A], opinion[B]
                opinion[A] = oA + beta * (oB - oA)
                opinion[B] = oB + beta * (oA - oB)
        # 在散点图中绘制当前时间步的意见状态
        axe[1].scatter([i] * MAX_PERSON, opinion, c='red')
        axe[1].set_ylabel('Opinion')

    # 在直方图中绘制意见分布
    axe[0].hist(opinion, bins=10, color='blue')
    axe[0].set_xlabel('Opinion')
    plt.show()

def test_defuant():
    """
    测试Deffuant模型的函数，通过不同的参数组合来观察模型行为。
    """
    defuant(0.5, 0.5)
    defuant(0.1, 0.5)
    defuant(0.5, 0.1)
    defuant(0.1, 0.2)
    plt.show()

def main():
    """
    主函数，处理命令行参数并运行Defuant模型。
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-beta', type=float, default=0.2, help='Coupling coefficient (beta)')
    parser.add_argument('-threshold', type=float, default=0.2, help='Opinion difference threshold')
    parser.add_argument('-defuant', action='store_true', help='Run the Deffuant model')
    parser.add_argument('-test_defuant', action='store_true', help='Run the test function for the Deffuant model')

    args = parser.parse_args()

    if args.defuant:
        defuant(args.beta, args.threshold)
    elif args.test_defuant:
        test_defuant()

if __name__ == '__main__':
    main()
```

按照上述步骤和示例运行代码，你应该能够成功地模拟Defuant模型并观察到意见演化的过程。
