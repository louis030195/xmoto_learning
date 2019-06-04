#!/usr/bin/python
# -*- coding: latin-1 -*-

from __future__ import division, print_function, unicode_literals

# Handle arguments (before slow imports so --help can be fast)
import argparse
parser = argparse.ArgumentParser(
    description="Train a DQN net for Xmoto")

# hparams
parser.add_argument("-nl", "--next-level", dest='next_level', type=int, default=200, help="number of steps to go next level")
parser.add_argument("-bc", "--behaviour-cloning", dest='behaviour_cloning',action='store_true', default=False, help="register yourself playing ?")
parser.add_argument("-m", "--model-dir", dest='model_dir', type=str, default='./model.pth', help="model directory")
parser.add_argument("-r", "--no-resume", dest='no_resume', action='store_true', default=False, help="start from existing model")
parser.add_argument("-e", "--eval", dest='eval', action='store_true', default=False, help="run inference")
parser.add_argument("-u", "--ugly", dest='ugly', action='store_true', default=False, help="ugly mode, easier to learn")



args = parser.parse_args()

import sys
import gym
import gym_xmoto
import math
import random
import numpy as np
from collections import namedtuple, deque
from itertools import count
from PIL import Image
from utils import save_gradient_images
import keyboard

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

env = gym.make('Xmoto-v0')
env.render(accelerated=not args.behaviour_cloning, ugly_mode=args.ugly) # Behaviour cloning ON = accelerated off

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class VanillaBackpropVisualization():
    """
        Produces gradients generated with vanilla back propagation from the image
    """
    def __init__(self, model):
        self.model = model
        self.gradients = None
        # Put model in evaluation mode
        self.model.eval()
        # Hook the first layer to get the gradient
        self.hook_layers()

    def hook_layers(self):
        def hook_function(module, grad_in, grad_out):
            self.gradients = grad_in[0]
            #print(grad_out[0])

        # Register hook to the first layer
        first_layer = list(self.model._modules.items())[0][1]
        first_layer.register_backward_hook(hook_function)

    def generate_gradients(self, input_image, action):
        # Forward
        model_output = self.model(input_image)
        # Zero grads
        self.model.zero_grad()
        # Target for backprop
        one_hot_output = torch.FloatTensor(1, model_output.size()[-1]).zero_()
        one_hot_output[0][action] = 1
        # Backward pass
        model_output.backward(gradient=one_hot_output)
        # Convert Pytorch variable to numpy array
        # [0] to get rid of the first channel (1,3,224,224)
        gradients_as_arr = self.gradients.data.numpy()[0]
        return gradients_as_arr

class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class DQN(nn.Module):

    def __init__(self, h, w):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(4, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)
        
        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size = 5, stride = 2):
            return (size - (kernel_size - 1) - 1) // stride  + 1
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        linear_input_size = convw * convh * 32
        self.head = nn.Linear(linear_input_size, env.action_space.n)

    def forward(self, x):
        x = x.view((-1, 4, 150, 200)).float()
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))



env.reset()

BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10

policy_net = DQN(200, 150).to(device)
#vbp = VanillaBackpropVisualization(policy_net)
target_net = DQN(200, 150).to(device)
#target_net.load_state_dict(policy_net.state_dict())

#target_net.eval()

optimizer = optim.RMSprop(policy_net.parameters())
memory = ReplayMemory(10000)

if not args.no_resume:
    print('Resuming from model at path', args.model_dir)
    checkpoint = torch.load(args.model_dir)
    policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
    target_net.load_state_dict(checkpoint['target_net_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

steps_done = 0


def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(env.action_space.n)]], device=device, dtype=torch.long)


def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see http://stackoverflow.com/a/19343/3343043 for
    # detailed explanation).
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.uint8)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()

    return loss

num_episodes = 200
mean_reward = 0
keys_list = ["up", "left", "down", "right", "space"]
def append_input(i):
    if i in keys_list:
        # print('Key', i.name, 'pressed')
        inputs.append(i)

if args.behaviour_cloning:
    inputs = deque()
    print("Start recording inputs")
    keyboard.on_press(append_input) # Add the input to the queue
    keyboard.on_release(lambda e: inputs.pop()) # Remove last input pressed

try:
    for i_episode in range(num_episodes):
        # Initialize the environment and state
        state = torch.from_numpy(env.reset())
        for t in count():
            # Select and perform an action
            if not args.behaviour_cloning:
                action = select_action(state)
            else:
                action = torch.tensor([[inputs[0] if len(inputs) > 0 else len(keys_list)]]) # Cast to tensor

            next_state, reward, done, _ = env.step(action.item())

            # Reward by states difference => push to explore
            #print(env.observation_space.shape)
            distance_between_states = np.linalg.norm(state.numpy() - next_state) / np.linalg.norm(np.zeros(env.observation_space.shape) - np.full(env.observation_space.shape, 255))
            reward += (distance_between_states * 10) ** 2

            # Cast to tensor
            reward = torch.tensor([reward], device=device)

            # Cast to tensor
            next_state = torch.from_numpy(next_state)

            # Store the transition in memory
            memory.push(state, action, next_state, reward)

            # Move to the next state
            state = next_state

            # Perform one step of the optimization (on the target network)
            loss = optimize_model()
            if done:
                # Generate gradients & save
                # Doesn't work
                # save_gradient_images(vbp.generate_gradients(state, action), str(i_episode) + '_Vanilla_BP_color')

                mean_reward = (mean_reward + reward) / 2
                print('Episodes %d - mean_reward %3f - loss %3f' % (i_episode, mean_reward, loss if loss is not None else 0))
                #print('distance_between_states: ', distance_between_states)
                break
                    
        # Update the target network
        if i_episode % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())
except KeyboardInterrupt:
    pass

print('Saving model ...')

torch.save({
            'policy_net_state_dict': policy_net.state_dict(),
            'target_net_state_dict': target_net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
            }, args.model_dir)

env.close()