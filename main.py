from agents import actor_critic_agent, policy_gradient_agent
from agent_trainer import train_agent
from environments import time_environment
import numpy as np

"""
This module sets up a reinforcement learning system
    creates an agent
    creates an environment to train the agent in
    defines a reward function

The goal of the agent is to capture the highest value at every time-step
    while also obtaining at least on of every object over the course of an episode
The max reward is achieved by accomplishing these two things
"""

# this is our reward matrix
#   represents the value of choosing each object at a certain time
# (normally we would generate the reward matrix, but here we just explicitly created it...)

            # Altair Vega Deneb
data= np.array([[-3,  1, -3],
                [-3,  1, -3], # hour
                [ 1,  2,  1], #  |
                [ 1,  2,  1], #  |
                [ 2,  3,  2], #  |
                [ 2,  3,  2], #  v
                [ 3,  3,  2],
                [ 3,  3,  3],
                [ 3,  3,  3],
                [ 3,  2,  3],
                [ 3,  2,  3],
                [ 2,  1,  2],
                [ 1,  1,  2],
                [ 1,  1,  1],
                [-3, -3,  1],
                [-3, -3,  1]] )

# MAX TOTAL REWARD: 34


# create our environment
env = time_environment(data)
obs_space = env.get_obs_space()
action_space = env.get_action_space()

# create an agent and train it
my_agent = actor_critic_agent(obs_space, action_space, learning_rate = 1e-2)
print("Agent's direcotry: "+my_agent.get_directory())
train_agent(my_agent, env, rewards_discount = 0.7)

print("\nDone training!")


# display how to access training graphs from tensorboard
print("\nType 'tensorboard --logdir "+my_agent.get_directory()+"' into the terminal")
print("Access on web browser through link: localhost:6006")

