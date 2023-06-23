import argparse
import time
import numpy as np
from utils.make_env import make_env
from PIL import Image

# TODO: replace with your models
from agents.random.submission import Agents as RandomSampleAgents
from agents.random_network.submission import Agents as RandomNetworkAgents
from agents.MADDPG.submission import Agents as MADDPG
import os

def run(config):
    env = make_env(config.env_id, discrete_action=True)

    # TODO: replace with you own agent model
    #agents = RandomNetworkAgents(env.observation_space[0].shape[0], env.action_space[0].n)
    #agents = RandomSampleAgents()
    agents = MADDPG()
    agents.load_parameters( 'maddpg_check50000' )

    total_reward = 0.
    
    img_list = [ ]
    for ep_i in range(config.n_episodes):
        print("Episode %i of %i" % (ep_i + 1, config.n_episodes))
        trace_list = []

        obs = env.reset()
        
        trace_list.append( Image.fromarray( env.render('rgb_array')[0] ) )

        episode_reward = 0.
        for t_i in range(config.episode_length):
            calc_start = time.time()
            
            actions = agents.act(obs)
            
            #obs: 3*(18,)
            #actions 3*(5,)
            #print( actions )
            obs, rewards, dones, infos = env.step(actions)
            episode_reward += np.array(rewards).sum()
            calc_end = time.time()
            elapsed = (calc_end - calc_start) * 1000.0

            # the elapsed should not exceed 10ms per step
            print("Elapsed %f ms" % (elapsed))

            # add render result
            trace_list.append( Image.fromarray( env.render('rgb_array')[0] ) )
        total_reward += episode_reward/config.episode_length
        print("Episode reward: %.2f" % (episode_reward/config.episode_length))

        img_list.append( trace_list )
        
    print("Mean reward of %d episodes: %.2f" % (config.n_episodes, total_reward/config.n_episodes))

    env.close()

    return img_list


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_id", default="simple_spread", type=str)
    parser.add_argument("--n_episodes", default=100, type=int)
    parser.add_argument("--episode_length", default=25, type=int)
    config = parser.parse_args()

    img_list = run(config)
    idx = 0
    for i in range( config.n_episodes ):
        for j in range( len(img_list[i]) ):
            img = img_list[i][j]

            img.save('./photo/img_{}.png'.format( idx ) )
            idx += 1
    

    #os.system( 'ffmpeg -r 10 -f image2 -i ./photo/img_%d.png output.mp4' )