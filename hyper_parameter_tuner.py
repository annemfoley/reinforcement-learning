from agents import actor_critic_agent, policy_gradient_agent
from agent_trainer import train_agent
from environments import time_environment
import numpy as np
import datetime
import pickle


# get our data:

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



# give some options for our network's hyper-parameters

hidden_layers_list = [1,2]
hidden_nodes_list = [10,15,20,25]
activation_list = ["sigmoid","relu","tanh"]
kernel_init_list = ["variance_scaling", "random_normal", "random_uniform"]
bias_init_list = ["zeros", "ones", "variance_scaling"]
pg_scalar_list = [0.5, 1, 2, 3] # will directly influence loss, look at avg reward instead
value_scalar_list = [0.5, 1, 2, 3] # same ^
learning_rate_list = [1e-4, 1e-3, 1e-2]
optimizer_list = ["Adam", "GradientDescent"]



# give some options for our training hyper-parameters

pre_episodes_list = [0,20,50,100]
batch_size_list = [10,20,50]
e_list = [0, 0.1, 0.3]
e_discount_list = [1, 0.99, 0.9]
rewards_discount_list = [0.9, 0.99] # will influence loss , look at average_reward instead




# use this directory name to look at tensorboard
#   will be printed when you run the program
#   tensorboard should have all runs of the program if you use master directory
now = datetime.datetime.now()
date = str(now.month)+"-"+str(now.day)+"-"+str(now.year)+"_"+str(now.hour)+"-"+str(now.minute)+"-"+str(now.second)
master_directory = "/tmp/astroplan/"+date
print("Use this directory:", master_directory)




# test the hyper-parameter(s) we choose

env = time_environment(data)

obs_space = env.get_obs_space()
action_space = env.get_action_space()

for i, activation in enumerate(activation_list):
    print("\n\nRUN #"+str(i+1))
    print("Activation:",activation)
    
    directory = master_directory + "/"+ str(activation)
    print("Agnet "+str(i+1)+"'s directory: "+directory)

    new_agent = actor_critic_agent(obs_space, action_space, value_scalar = 0.5,
        activation = activation, directory = directory)
        
    train_agent(new_agent, env, rewards_discount = 0.7)



print("\nDone!")
print("\nType 'tensorboard --logdir "+master_directory+"' into the terminal")
print("Access on web browser through link: localhost:6006")
