from agents import actor_critic_agent, policy_gradient_agent
from environments import time_environment
import numpy as np

"""
Function to train an agent on a data-set

Many adjustable parameters for training the agent

Includes no use of tensorflow at all--should all be encapsulated with the agent
"""

def train_agent(agent, env,
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
                ):
    """
    trains the agent in an environment
    """

    # make sure our environment is reset
    env.reset()
    
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
            done = False
                    
            while done == False:
                # randomly choose an action
                action = np.random.randint(0,env.get_action_space())
                state, reward, done = env.take_action(action)
                states.append([state])
                episode_rewards.append(reward)

            # discount our rewards
            episode_rewards = discount(episode_rewards, rewards_discount)

            # update our state value
            agent.update_state_values(states, episode_rewards)

    
    
    # ------------- start training -------------------


    # initialize our lists to contain batch data
    states, actions, rewards = [], [], []

    # make sure our environment is reset
    env.reset()

    for episode in range(episodes):

        # every episode, keep track of our rewards gained from the 'data' matrix as well as our final reward
        episode_rewards = []
        diversity_rewards = []

        # get our initial state
        state = env.get_first_state()
        done = False

        # for display purposes:
        my_actions = np.zeros(shape = [env.get_obs_space()],dtype = np.int32)
        display_reward = 0

        while done == False:

            # get our next action, with a chance of exploring
            if np.random.random() < e and display == True and episode % display_rate==0:
                # choose a random action
                action = np.random.randint(0,env.get_action_space())
                e*=e_discount
            else:
                # get what our agent thinks is the best action
                action = agent.get_action(state)

            # update our gradients if it is a policy-gradient agent
            if isinstance(agent, policy_gradient_agent):
                agent.update_gradients(state, action)
            
            # pass the action through the environment
            state, reward, done = env.take_action(action)

            # for display purposes:
            total_reward += reward
            display_reward += reward
            my_actions[state] = action

            # update our value reward and final reward (which is 0 unless it's the end of the episode)
            episode_rewards.append(reward)

            # add values to our batch information
            states.append([state])
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
            agent.write_summaries([[state]], [action], [episode_rewards[-1]], average_reward, summary_number)
            summary_number += 1


        if display == True and episode % display_rate==0:
            # display our information
            print("\nEpisode #"+str(episode))
            print("Actions taken:\t" , my_actions)
            print("Episode reward:\t",display_reward)
            print("Average reward:\t" , average_reward)

