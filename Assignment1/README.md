# 实践作业一：动态规划、蒙特卡洛与时序差分方法


## 概述

本次实践作业将在以下环境进行：

![mini_grid.png](https://github.com/joenghl/SYSU_2023SpringRL/blob/master/docs/images/mini_grid.png?raw=true)

该环境由一个 6×6 网格组成，其中黄色圆圈为智能体出发点，黑色格子为无法通过的墙壁，若智能体向着墙壁方向移动，则会停留在原地，地图边界的移动同理。带有黄色边框的格子为终止状态，即若智能体行动至此状态则整个 episode 结束。红色和绿色表示当前该状态的奖励值，奖励越高则绿色越深，奖励越低则红色越深。

状态空间：36（`0-35` 这 36 个整数，左下角为 0，向右 +1，向上 +6）

动作空间：4（`0-3` 这 4 个整数分别代表左、右、上、下）

奖励函数：普通格子 -0.1，到达终点(s=9) +1.0，掉入陷阱(s=23) -1.0。

## 环境配置

整个项目使用 Python 语言，在配置环境前，推荐使用 `conda` 工具管理 Python 编程环境。

#### 1. 安装 `conda` 工具 (推荐, 可选)

推荐从 [Miniconda Official](https://docs.conda.io/en/latest/miniconda.html) 下载 `Miniconda` ，安装时勾选自动添加环境变量选项，若没有勾选则需要手动添加，其他选项默认即可。

#### 2. 新建并激活用于此项目的虚拟环境 

**Option a. 使用 `conda` :**

```shell
conda create -n 2023RL python=3.9
conda activate 2023RL
```

**Option b. 使用 Python 工具 :**

```shell
python -m venv 2023RL
source 2023RL/bin/activate
```

#### 3. 配置项目仓库

```shell
git clone https://github.com/joenghl/SYSU_2023SpringRL.git
cd SYSU_2023SpringRL
```

#### 4. 安装依赖库

```shell
python -m pip install numpy==1.21.2 gym==0.10.0 pyglet==1.2.4
```

## 任务一：动态规划方法

在动态规划方法中，无需考虑智能体和环境的交互过程，只需利用奖励函数和折扣因子计算每个状态的 V 值。

请从 `策略迭代` 或者 `价值迭代`  中选择一种方法实现，`Assignmen1/dp.py` 提供了一个基础的模板，在此基础上完成你的代码，模板中内容可以根据需要更改，但需要保证其是一个可运行的程序，运行结果需打印出最终收敛的 V 表 `v` ，以及使用 `env.update_r(v)` 后将 V 表的值同步至环境端后的渲染图形 (模板中已给出渲染方法)，输出格式不限制。

随机策略下评估收敛渲染图示例：

![dp.png](https://github.com/joenghl/SYSU_2023SpringRL/blob/master/docs/images/dp.png?raw=true)

## 任务二：蒙特卡洛方法

需要智能体和环境交互，交互方式可通过以下方式实现：

```python
env = MiniGrid()  # 初始化环境
state = env.reset()  # 重置环境状态
done = False
t = 0
while not done and t < env.max_step:
    t += 1
    action = act(state)  # 根据当前策略和状态采取动作
    next_state, reward, done, info = env.step(action)  # 和环境交互
    state = next_state
```

蒙特卡洛方式在每次 episode 结束后（即从每次 `env.reset()`  到一个回合结束，结束可能有两个原因，一是环境 `step` 后返回 `done=True` ，二是当前回合步数达到最大值 `t>=env.max_step()`）更新遍历过的状态的 V 值。

`Assignment1/mc.py` 提供了一个示例模板，模板内容可根据需求使用和修改，但需要保证其是一个可运行的程序，运行结果需打印出最终收敛的 V 表 `v` ，以及使用 `env.update_r(v)` 后将 V 表的值同步至环境端后的渲染图形，输出格式不限制。

提示：MC 和 TD 方法需要运用探索和 Q 表技巧。

## 任务三：时序差分方法

需要智能体和环境交互，交互方式可参考 `蒙特卡洛` 方法中的描述。时间差分方法不需要等到整个 episode 结束后再更新，而是每步都可以更新 V 值（TD(0)方法）。

`Assignment1/td.py` 提供了一个示例模板，模板内容可根据需求使用和修改，但需要保证其是一个可运行的程序，运行结果需打印出最终收敛的 V 表 `v` ，以及使用 `env.update_r(v)` 后将 V 表的值同步至环境端后的渲染图形，输出格式不限制。

## 提交

### 代码

提交三个 `.py` 文件，前缀：学号\_姓名，后缀：\_dp, \_mc, _td 分别对应以上三个任务，若提交后需更新代码，请加后缀 \_v1, \_v2，以此类推。

如: 22000000\_张三\_dp.py, 22000000\_张三\_mc.py, 22000000\_张三\_td.py。

### 报告

提交一份 `.pdf` 文件，命名：学号\_姓名，该报告包需含本次作业的三个任务，报告内容可自由发挥，详略自定。可描述核心代码的思路等。

因代码运行的不确定性，可将代码执行结果（V 表、收敛后的渲染图形等）在报告中体现。

### 提交方式

将以上代码和报告文件分别上传至 FTP 服务器根目录下的 `研究生强化学习作业1_code` 和 `研究生强化学习作业1_report`，截止日期: 2023.4.28 23:59。

IP: 222.200.177.152

Port: 1021

User: ftpstu

Password: 123456

Tips: Windows 下在文件夹路径填入 `ftp://222.200.177.152:1021/` 后输入用户名和密码即可连接至 FTP 服务器（校园网内网）。
