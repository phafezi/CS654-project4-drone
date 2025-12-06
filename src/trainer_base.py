import numpy as np
from itertools import count
from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from gym_pybullet_drones.utils.utils import sync, str2bool
from gym_pybullet_drones.utils.enums import (
    DroneModel,
    Physics,
    ActionType,
    ObservationType,
)
from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.envs.NavigationAviary import NavigationAviary

from src.ActorCritic import ActorCriticModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

gamma = 0.99
SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])
saved_actions = []
rewards = []         

def select_action(state, model):
    img = torch.from_numpy(state["rgb"]).float().to(device)
    kin = torch.from_numpy(state["kin"]).float().to(device) 

    dist, value = model(img, kin)

    action = dist.sample()                      
    log_prob = dist.log_prob(action).sum(dim=-1)

    # save to action buffer
    saved_actions.append(SavedAction(log_prob, value))
    
    #get the action numpy array for each rotor clipped between -1 to 1 according to the action sample space
    action = action.detach().cpu().numpy()
    action = np.clip(action, -1.0, 1.0)
    return action


def finish_episode(model, optimizer):
    """
    Training code. Calculates actor and critic loss and performs backprop.
    """
    R = 0
    policy_losses = [] # list to save actor (policy) loss
    value_losses = [] # list to save critic (value) loss
    returns = [] # list to save the true values

    # calculate the true value using rewards returned from the environment
    for r in rewards[::-1]:
        R = r + gamma * R
        returns.insert(0, R)

    returns = torch.tensor(returns,  dtype=torch.float32, device=device)

    for (log_prob, value), R in zip(saved_actions, returns):
        advantage = R - value.item()

        # calculate actor (policy) loss
        policy_losses.append(-log_prob * advantage)


        target = torch.tensor([R], dtype=torch.float32, device=device)
        target = target.view(-1, 1)
        # calculate critic (value) loss using L1 smooth loss
        value_losses.append(F.smooth_l1_loss(value, target))

    # reset gradients
    optimizer.zero_grad()

    # sum up all the values of policy_losses and value_losses
    loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()

    # perform backprop
    loss.backward()
    optimizer.step()

    # reset rewards and action buffer
    del rewards[:]
    del saved_actions[:]


def main():
    running_reward = 0

    env = NavigationAviary(
        drone_model=DroneModel.CF2X,
        initial_xyzs=np.array([[0.0, 0.0, 1.0]]),
        initial_rpys=np.zeros((1, 3)),
        physics=Physics.PYB,
        pyb_freq=240,
        ctrl_freq=48,
        gui=False,
        record=False,
        obs=ObservationType.JOINT,
        act=ActionType.RPM,
    )

    state, _ = env.reset()

    in_channels = state["rgb"].shape[-1]     
    kin_dim = state["kin"].shape[-1]         
    n_actions = 4                             
    model = ActorCriticModel(in_channels=in_channels, n_actions=n_actions, kin_dim=kin_dim).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=3e-4)

    # run infinitely many episodes
    for i_episode in count(1):

        # reset environment and episode reward
        state, _ = env.reset()
        ep_reward = 0

        # for each episode, only run 9999 steps so that we don't
        # infinite loop while learning
        for t in range(1, 10000):

            # select action from policy
            action = select_action(state, model)

            # take the action
            state, reward, terminated, truncated, _ = env.step(action)

            rewards.append(reward)
            ep_reward += reward
            if terminated or truncated:
                break

        # update cumulative reward
        running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward

        # perform backprop
        finish_episode(model, optimizer)

        # log results
        if i_episode % 100 == 0:
            print(f'Episode {i_episode}\tLast reward: {ep_reward:.2f}\tAverage reward: {running_reward:.2f}')



if __name__ == '__main__':
    main()