import gym
from numpy import number
from stable_baselines3.ppo import PPO
import torch.nn as nn
import argparse

from learners import *
from utils import *

import os
import stat

def make_env():
    # return gym.make("CartPole-v1")
    return gym.make("LunarLander-v2")

def get_expert():
    # model = PPO.load("experts\CartPole-v1\cartpole_expert")
    # model = PPO.load("ppo1_cartpole")

    model = PPO.load("experts\LunarLander-v2\lunarlander_expert")
    return model

def get_expert_performance(env, expert):
    Js = []
    for _ in range(100):
        obs = env.reset()
        J = 0
        done = False
        hs = []
        while not done:
            action, _ = expert.predict(obs)
            obs, reward, done, info = env.step(action)
            hs.append(obs[1])
            J += reward
        Js.append(J)
    ll_expert_performance = np.mean(Js)
    return ll_expert_performance

def trajectories(number_of_trajectories, function, network):
    states, actions = function(network)
    states = np.array(states); actions = np.array(actions)
    if number_of_trajectories>1:
        for i in range(number_of_trajectories-1):
            s, a = function(network)
            states = np.concatenate((states,np.array(s)), axis=0); actions = np.concatenate((actions,np.array(a)), axis=0)

    return states, actions

def main(args):
    env = make_env()
    expert = get_expert()
    
    performance = get_expert_performance(env, expert)
    print('=' * 20)
    print(f'Expert performance: {performance}')
    print('=' * 20)
    
    # net + loss fn
    if args.truncate:
        net = create_net(input_dim=6, output_dim=4)
        # net = create_net(input_dim=4, output_dim=2)
    else:
        net = create_net(input_dim=8, output_dim=4)
        # net = create_net(input_dim=4, output_dim=2)
    
    loss_fn = nn.CrossEntropyLoss()
    lr = 0.001
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    
    if args.bc:
        # TODO: train BC
        # Things that need to be done:
        # - Roll out the expert for X number of trajectories (a standard amount is 10).
        # - Create our BC learner, and train BC on the collected trajectories.
        # - It's up to you how you want to structure your data!
        # - Evaluate the argmax_policy by printing the total rewards.
        
        # --- Rollout from Expert ---
        number_of_trajectories = 10
        function = lambda network: expert_rollout(network, env, False)
        states, actions = trajectories(number_of_trajectories, function, expert)
        
        # --- Creating BC Learner ---
        # net = training(net, loss_fn, optimizer, states, actions)
        BCobject = BC(net, loss_fn); net = BCobject.learn(env, states, actions)
        
        # --- Assessment ---
        reward = eval_policy(argmax_policy(net), env, False)
        print('Expert Reward (BC)', performance); print('Learner Reward (BC)', reward)

    else:
        # TODO: train DAgger
        # Things that need to be done.
        # - Create our DAgger learner.
        # - Set up the training loop. Make sure it is fundamentally interactive!
        # - It's up to you how you want to structure your data!
        # - Evaluate the argmax_policy by printing the total rewards.
        
        # --- DAgger Learner ---
        DAggerobject = DAgger(net, loss_fn, expert); net = DAggerobject.learn(env)
        
        # --- Final Reward ---
        reward = eval_policy(argmax_policy(net), env, False)
        print('Expert Reward (DAgger)', performance); print('Learner Reward (DAgger)', reward)
        
def get_args():
    parser = argparse.ArgumentParser(description='imitation')
    parser.add_argument('--bc', action='store_true', help='whether to train BC or DAgger')
    parser.add_argument('--n_steps', type=int, default=10000, help='number of steps to train learner')
    parser.add_argument('--truncate', action='store_true', help='whether to truncate env')
    
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    main(get_args())