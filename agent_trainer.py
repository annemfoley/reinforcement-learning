from agents import actor_critic_agent, policy_gradient_agent
import numpy as np
import matplotlib.pyplot as plt




def step(data, timestep, action, object_list, final_reward_subtraction):
    """
    this function is essentially our environment,
        it updates our state (not the actual network input, just what actions we have aquired to this point)
        it returns a reward based on the number in the 'data' matrix
        it also returns a final reward based on diversity of objects
    """

    # get the reward for the value in the 'data' matrix
    # we're going to call  this VALUE REWARD
    reward = data[timestep][action]

    # update our list of what objects we do and don't have as of now in this episode
    if object_list[action]==0:
        object_list[action] = 1

    final_reward = 0
    # calculate our final reward based on diversity only when the episode ends
    # we're going to call this DIVERSITY REWARD
    if timestep+1 == len(data):
        final_reward = 0
        for val in object_list:
            if val == 0:
                final_reward -= final_reward_subtraction # deduct a certain value for every object not ever chosen

    return object_list, reward, final_reward





"""
Function to train an agent on a data-set

The data-set should be a 2D array:
    row # represents the timestep
    column # represents the action
    number in [i,j] is the value of choosing action j at timestep i

    e.g.:
    would represent a 3-timestep progression with 4 possible actions
    data = [[ _, _, _, _],
            [ _, _, _, _],
            [ _, _, _, _]]

Many adjustable hyper-parameters for training the agent

Includes no use of tensorflow at all--should all be encapsulated with the agent
"""

def train_agent(agent, data,
                pre_train = True,           # whether or not to pre-train state values
                display = True,             # whether or not to display/print
                plot = True,                # whether or not to plot to tensorboard
                pre_episodes = 10,          # how many pre-training episodes to conduct
                episodes = 500,             # how many episodes we want to run
                batch_size = 10,            # number of episodes per update for our agent
                display_rate = 50,          # how often to display/print our status
                plot_rate = 10,             # how often to write to tensorboard
                e = 0.1,                    # fraction of time for exploration (taking random moves)
                e_discount = 0.9,           # how much to discount our exploration rate every time we explore
                rewards_discount = 0.9,      # how much to discount our rewards by
                final_reward_subtraction = 2 # how much to subtract for each object we didn't pick at the end of an episode
                ):
    """
    simulates an environment given data and trains the agent
    """

    # specify our action-space and timesteps
    action_space = len(data[0])
    max_timesteps = len(data)

    total_reward = 0
    summary_number = 0

    
    def discount(array, discount_rate):
        """
        'discounts' an array of values:
                adds a fraction of every following value to the preceding value
                mostly needed to discount any reward earned at the end for diversity
                this will make the agent aware of how current actions affect future rewards

        e.g., discount( [0,0,0,1], 0.5) = [0.125, 0.25, 0.5, 1]           
        """
        discounted_array = np.empty(len(array))
        cumulative = 0
        for i in reversed(range(len(array))):
            cumulative = array[i] + cumulative * discount_rate
            discounted_array[i] = cumulative
        return discounted_array




    # ------------- make some pre-training state estimates ------------

    # only if it's a actor-critic agent

    # part of our loss is the estimation of a state's value
    # we want to tune this slightly before we actually start training the policy
    if pre_train == True and isinstance(agent, actor_critic_agent)==True:

        for episode in range(pre_episodes):
            
            states, episode_rewards = [], []
            my_objects = np.zeros(shape = [action_space], dtype = np.int32)
            
            for timestep in range(max_timesteps):
                # randomly choose an action
                action = np.random.randint(0,action_space)
                my_objects, reward, final_reward = step(data, timestep, action,
                                                        my_objects, final_reward_subtraction)
                states.append([timestep])
                episode_rewards.append(reward)

            # discount our rewards
            episode_rewards = discount(episode_rewards, rewards_discount)

            # update our state value
            agent.update_state_values(states, episode_rewards)

    
    

    # ------------- start training -------------------


    # initialize our lists to contain batch data
    states, actions, rewards = [], [], []

    for episode in range(episodes):

        # every episode, keep track of our rewards gained from the 'data' matrix as well as our final reward
        episode_rewards = []
        diversity_rewards = []

        # keep track of what objects we have obtained this episode
        my_objects = np.zeros(shape=[action_space],dtype = np.int32)

        # for display purposes:
        my_actions = np.zeros(shape = [max_timesteps],dtype = np.int32)
        display_reward = 0

        for timestep in range(max_timesteps):

            # get our next action, with a chance of exploring
            if np.random.random() < e:
                # choose a random action
                action = np.random.randint(0,action_space)
                e*=e_discount
            else:
                # get what our agent thinks is the best action
                action = agent.get_action(timestep)

            # update our gradients if it is a policy-gradient agent
            if isinstance(agent, policy_gradient_agent):
                agent.update_gradients(timestep, action)
            
            # pass the action through the environment
            my_objects, reward, final_reward = step(data, timestep, action, my_objects, final_reward_subtraction)

            # for display purposes:
            total_reward += reward+final_reward
            display_reward += reward+final_reward
            my_actions[timestep] = action

            # update our value reward and final reward (which is 0 unless it's the end of the episode)
            episode_rewards.append(reward+final_reward)

            # add values to our batch information
            states.append([timestep])
            actions.append(action)
            
        # calculate our average reward so tensorboard can plot it
        average_reward = round(total_reward/(episode+1),3)

        # discount our diversity rewards and add that to our value rewards
        episode_rewards = discount(episode_rewards, rewards_discount)
        rewards.append(episode_rewards)

        if episode % batch_size == 0 and episode != 0:
            # update our batch
            agent.update_batch(states, actions, rewards)
            states, actions, rewards = [], [], []
            
        if plot == True and episode % plot_rate == 0 and episode != 0:
            # write summaries to tensorboard, make sure shapes are correct
            agent.write_summaries([[timestep]], [action], [episode_rewards[-1]], average_reward, summary_number)
            summary_number += 1


        if display == True and episode % display_rate==0:
            # display our information
            print("\nEpisode #"+str(episode))
            print("Actions taken:\t" , my_actions)
            print("Episode reward:\t",display_reward)
            print("Average reward:\t" , average_reward)

