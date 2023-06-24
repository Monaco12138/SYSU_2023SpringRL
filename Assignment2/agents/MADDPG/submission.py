import torch
import torch.nn.functional as F
import numpy as np
from utils.make_env import make_env
import os


def onehot_from_logits(logits, eps=0.01):
    ''' 生成最优动作的独热（one-hot）形式 '''
    argmax_acs = (logits == logits.max(1, keepdim=True)[0]).float()
    # 生成随机动作,转换成独热形式
    rand_acs = torch.autograd.Variable(torch.eye(logits.shape[1])[[
        np.random.choice(range(logits.shape[1]), size=logits.shape[0])
    ]],
                                       requires_grad=False).to(logits.device)
    # 通过epsilon-贪婪算法来选择用哪个动作
    return torch.stack([
        argmax_acs[i] if r > eps else rand_acs[i]
        for i, r in enumerate(torch.rand(logits.shape[0]))
    ])


def sample_gumbel(shape, eps=1e-20, tens_type=torch.FloatTensor):
    """从Gumbel(0,1)分布中采样"""
    U = torch.autograd.Variable(tens_type(*shape).uniform_(),
                                requires_grad=False)
    return -torch.log(-torch.log(U + eps) + eps)


def gumbel_softmax_sample(logits, temperature):
    """ 从Gumbel-Softmax分布中采样"""
    y = logits + sample_gumbel(logits.shape, tens_type=type(logits.data)).to(
        logits.device)
    return F.softmax(y / temperature, dim=1)


def gumbel_softmax(logits, temperature=1.0):
    """从Gumbel-Softmax分布中采样,并进行离散化"""
    y = gumbel_softmax_sample(logits, temperature)
    y_hard = onehot_from_logits(y)
    y = (y_hard.to(logits.device) - y).detach() + y
    # 返回一个y_hard的独热量,但是它的梯度是y,我们既能够得到一个与环境交互的离散动作,又可以
    # 正确地反传梯度
    return y

class TwoLayerFC(torch.nn.Module):
    def __init__(self, num_in, num_out, hidden_dim):
        super().__init__()
        self.fc1 = torch.nn.Linear(num_in, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, num_out)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


ENV = make_env('simple_spread', discrete_action=True)
N_AGENT = 3
N_ACTION = ENV.action_space[0].n  # 5
N_OBS = ENV.observation_space[0].shape[0]  # 18

class DDPG:
    ''' DDPG算法 '''
    def __init__(self, state_dim, action_dim, hidden_dim, device):
        
        self.actor = TwoLayerFC(state_dim, action_dim, hidden_dim).to(device)
        

    def take_action(self, state, explore=False):
        action = self.actor(state)
        if explore:
            action = gumbel_softmax(action)
        else:
            action = onehot_from_logits(action)
        return action.detach().cpu().numpy()[0]

class Agents:
    def __init__( self, agent_num=N_AGENT, hidden_dim=64 ):
        
        self.agent_num = agent_num
        #self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = "cpu"

        self.agents = []
        for i in range( agent_num ):
            self.agents.append(
                DDPG(N_OBS, N_ACTION, hidden_dim, self.device))
        
        self.critic_criterion = torch.nn.MSELoss()

        self.load_parameters( 'maddpg_check200000' )

    def act(self, states, explore=False):
        states = [
            torch.tensor([states[i]], dtype=torch.float, device=self.device)
            for i in range( self.agent_num )
        ]
        return [
            agent.take_action(state, explore)
            for agent, state in zip(self.agents, states)
        ]

    def load_parameters( self, save_name ):
        for i in range( len(self.agents) ):
            self.agents[i].actor.load_state_dict( torch.load(os.path.join( '../Assignment2/agents/MADDPG', '{}_agent{}.pth'.format(save_name, i)) ) )
            