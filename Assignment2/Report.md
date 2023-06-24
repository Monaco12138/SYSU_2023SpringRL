# 多智能体强化学习实验报告

柴百里，22214414



## MADDPG

本次我使用多智能体 DDPG（muli-agent DDPG，MADDPG）算法来解决多智能体粒子的协作任务。

在MADDPG中，所有智能体共享一个中心化的 Critic 网络，该 Critic 网络在训练的过程中同时对每个智能体的 Actor 网络给出指导，而执行时每个智能体的 Actor 网络则是完全独立做出行动，即去中心化地执行。 

参考如下的文章来进行实验

>  Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments 

如图所示
<img src=".\result\framework.png" alt="framework" style="zoom:80%;" />

每个智能体用 Actor-Critic 的方法训练，但不同于传统单智能体的情况，在 MADDPG 中每个智能体的 Critic 部分都能够获得其他智能体的策略信息。具体来说，考虑一个有 N 个智能体的场景，每个智能体的策略参数为 
$$
\theta = \{ \theta_1, \theta_2, ..., \theta_N \}
$$

记
$$
\pi = \{ \pi_1, \pi_2, ..., \pi_N \}
$$

为所有智能体的策略集合，那么我们可以写出在随机性策略情况下每个智能体的期望收益的策略梯度：
$$
\triangledown_{\theta_i}J(\theta_i) = \mathbb{E}_{ s \sim p^{\mu}, a \sim\pi_i}[\triangledown_{\theta_i}\log\pi_i(a_i|o_i)Q^\pi_i(x,a_1,...,a_N) ]
$$

其中， 
$$
Q^\pi_i(x,a_1,...,a_N)
$$

就是一个中心化的动作价值函数。

对于确定性策略来说，考虑现在有N个连续的策略，可以得到DDPG的梯度公式:
$$
\triangledown_{\theta_i}J(\mu_i) = \mathbb{E}_{ x \sim D}[\triangledown_{\theta_i}\mu_i(o_i) \triangledown_{a_i}Q^\mu_i(x, a_1,...,a_N)|_{a_i=\mu_i(o_i)}]
$$

其中，D 是我们用来存储数据的经验回放池，它存储的每一个数据为
$$
(x,x',a_1,...,a_N,r_1,...,r_N)
$$

------

__MADDPG的算法流程如下：__ 
- 随机初始化每个智能体的 Actor 网络和 Critic 网络
- **for** 序列 **do**
	- 初始化一个随机过程，用于动作探索；
	- 获取所有智能体的初始观测 ；
	- **for** **do**：
		- 对于每个智能体 __i__，用当前的策略选择一个动作；
		- 执行动作 并且获得奖励和新的观测；
		- 把 存储到经验回放池中；
		- 从中随机采样一些数据;
		- 对于每个智能体 __i__，中心化训练 Critic 网络
		- 对于每个智能体 __i__，训练自身的 Actor 网络
		- 对每个智能体 __i__，更新目标 Actor 网络和目标 Critic 网络
	- **end for**
- **end for**

------

__文件结构如下：__

注释为新增文件

```python
Assignment2
│   README.md
│   run_test.py  
│   Report.md   #实验报告
│   MADDPG.py   # MADDPG 训练代码
│	output.mp4  # 渲染后的视频
│
└───agents  
│   │ 
│   └─random  
│   │   submission.py
│   │  
│   └─random_network  
│   │   │   agent1.pth
│   │   │   agent2.pth
│   │   │   agent3.pth
│   │   │   submission.py
│	│
│	│
│	└─MADDPG	#MADDPG 测试代码
│		│	maddpg_check200000_agent0.pth	#训练好的三个模型
│		│	maddpg_check200000_agent1.pth	#
│		│	maddpg_check200000_agent2.pth	#
│		│	submission.py					#
│
└───utils
    |   make_env.py   
```



## 实验结果



__训练效果图：__

训练过程中的 平均Reward 随训练轮次的 变化如图。随着训练轮数的提高，平均Reward 趋于-4.8 左右

<img src=".\result\RL.png" alt="RL" style="zoom:80%;" />



__测试结果:__

1. Reward

   <img src=".\result\reward.png" alt="image-20230624161414414" style="zoom:80%;" />

   测试Reward如图，可以看到每一轮的Reward 在 __-5.5__  到 __-4.5__ 之间浮动，

   测试100轮最终的平均Reward为 __-4.79__ 

   

2. Running time

   <img src=".\result\time.png" alt="image-20230624161414414" style="zoom:80%;" />
   
   如图显示每一步的测试用时(__单位ms__)，可以看到每步耗时基本在 __2ms__ 到 __3ms__ 直接。



3. 渲染视频，见附件

