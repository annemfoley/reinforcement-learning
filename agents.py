import numpy as np
import tensorflow as tf
import datetime


"""
Agent Super-Class:

Provides a foundation for other agent classes
    - meant to 

Requires implementation of get_action(), update_batch(), create_summaries(), and write_summaries()
"""


class agent:


    def __init__(self,directory=None):
        
        now = datetime.datetime.now()
        date = str(now.month)+"-"+str(now.day)+"-"+str(now.year)+"_"+str(now.hour)+"-"+str(now.minute)+"-"+str(now.second)
        
        if directory==None:
            self.directory = "/tmp/astroplan/"+date
        else:
            self.directory = directory

        print("\nNew agent's personal directory: "+self.directory)

    def reset(self):
        """
        reset our default tensorflow graph and start a new session
        """
        tf.reset_default_graph()
        agent.sess = tf.Session()

    def new_dense_layer(self, nodes, kernel_init, bias_init, inputs):
        """
        creates a new layer
        given the nodes, weight initializer, bias initializer, and previous (input) layer
        """
        layer_nodes = tf.layers.Dense(nodes, activation = tf.nn.sigmoid, kernel_initializer = kernel_init, bias_initializer = bias_init)
        layer = layer_nodes.apply(inputs)
        return layer

    def get_directory(self):
        """
        return the directory as a string
        """
        return self.directory

    def get_action(self):
        raise NotImplementedError

    def update_batch(self):
        raise NotImplementedError

    def create_summaries(self):
        raise NotImplementedError

    def write_summaries(self):
        raise NotImplementedError

    












"""
Actor-Critic Agent

Consists of a 2-part graph:
    input to first part is the state
    will generate an action (simply the output with the max value)

    input to second part is the state and the action taken
    will generate a perceieved value of the state-action pair, goal is to accurately predict values
    
Losses:
    includes a policy-gradient loss that can be thought of as the network's confidence
    also includes a value loss that is the difference between the predicted value and the actual value (the discounted reward)

Also writes the losses, average reward, weights, and biases to tensorboard

Agent is initialized with adjustable hyper-parameters for the network

"""



class actor_critic_agent(agent):

    def __init__(self, obs_space, action_space,
                 one_hot = True,                # whether or not to encode the state
                 hidden_layers = 1,             # the number of hidden layers
                 hidden_nodes = 20,             # the nodes per hidden layer
                 kernel_init = "variance_scaling",# the weights initializer for each layer
                 bias_init = "zeros",           # the bias initailizer for each layer
                 pg_scalar = 1,                 # magnitude of policy gradient loss
                 value_scalar = 1,              # magnitude of value loss
                 learning_rate = 1e-2,          # learning rate for optimizer
                 optimizer = "Adam",            # type of optimizer
                 directory = None):             # directory name for tensorboard

        """
        creates an agent with a network and training procedure
        """

        super().__init__(directory = directory)

        super().reset()
        
        kernel_init_dict = {"variance_scaling": tf.initializers.variance_scaling,
                            "random_normal": tf.initializers.random_normal,
                            "random_uniform": tf.initializers.random_uniform}
        bias_init_dict = {"zeros": tf.initializers.zeros,
                          "ones": tf.initializers.ones,
                          "variance_scaling": tf.initializers.variance_scaling}

        self.obs_space = obs_space # the number of states
        self.action_space = action_space # the number of actions (output)

        # set all our hyper-parameters
        self.hidden_layers = hidden_layers
        self.hidden_nodes = hidden_nodes
        self.kernel_init = kernel_init_dict[kernel_init]
        self.bias_init = bias_init_dict[bias_init]
        self.pg_scalar = pg_scalar
        self.value_scalar = value_scalar
        self.lr = learning_rate

        # set our optimizer with given name and learning rate
        op_dict = {"Adam": tf.train.AdamOptimizer(self.lr),
                   "Gradient Descent": tf.train.GradientDescentOptimizer(self.lr)}
        self.op = op_dict[optimizer]
        
        # create our summary writer
        self.writer = tf.summary.FileWriter(self.directory)



        # ------------start building our network-------------
        
        #   name scopes are used for the tensorboard graph


        # our input consists of our state
        #   change to a one-hot format to make it easier for our network to recognize
        with tf.name_scope("network_inputs"):
            self.state = tf.placeholder(shape = [None,1],dtype = tf.int32)
            self.state_one_hot= tf.one_hot(self.state, obs_space)

        # keep track of our previous layer to feed to the next layer
        previous= self.state_one_hot

        # create our hidden layers using the new_hidden_layer method
        for i in range(hidden_layers):
            with tf.name_scope("hidden"+str(i+1)):
                new_hidden_layer = super().new_dense_layer(self.hidden_nodes,self.kernel_init,self.bias_init,previous)
                previous = new_hidden_layer

        with tf.name_scope("outputs"):
            self.state_value = super().new_dense_layer(1,self.kernel_init,self.bias_init,previous)
            self.outputs = super().new_dense_layer(self.action_space,self.kernel_init,self.bias_init,self.state_one_hot)
            self.outputs = tf.squeeze(self.outputs)

        # -----------set up our network's training procedure------------


        # to calculate loss, we need the action we took and the reward we gained
        with tf.name_scope("back-prop_inputs"):
            self.action_holder = tf.placeholder(shape = [None], dtype = tf.int32)
            self.reward_holder = tf.placeholder(shape = [None], dtype = tf.float32)
            self.q_value = tf.nn.sigmoid(self.reward_holder)


        # calculate our losses:
        #   actor-critic method of loss
        #   policy-gradient loss: we want to calculate the gradients, then encourage/discourage based on the advantage of the action
        #       - is independent from actual discounted reward, or "q-value"
        #       - only is told based on the 'advantage' of the action which way to move along the gradient
        #   value loss: want our network to accurately understand the value of a function
        #       - the state-value is like a weighted average of the state-action pair values for that state
        #       - advantage of an action would be the state-action value - the state value
        
        with tf.name_scope("losses"):
            self.pg_loss = self.pg_scalar * tf.reduce_mean((self.q_value - self.state_value) *
                                tf.nn.sparse_softmax_cross_entropy_with_logits(logits = self.outputs, labels = self.action_holder), name = "pg_loss")
            self.value_loss = self.value_scalar * tf.reduce_mean(tf.square(self.q_value - self.state_value), name = "value_loss")
            self.total_loss = self.pg_loss + self.value_loss


        # now we define our gradients and optimizers
        #   have an optimizer for just the value-function, mostly for if we want to pre-train
        #   second optimizer for overall loss when we actually train
        #   gradients are only calculated through the variables they are dependent on
        with tf.name_scope("optimizer"):
            self.value_variables = tf.trainable_variables()[:-2]
            self.value_grads = tf.gradients(self.value_loss,self.value_variables)
            self.value_grads_and_vars = list(zip(self.value_grads, self.value_variables))
            self.value_update = self.op.apply_gradients(self.value_grads_and_vars)
                                                    
            self.grads = tf.gradients(self.total_loss, tf.trainable_variables())
            self.grads_and_vars = list(zip(self.grads, tf.trainable_variables()))
            self.update = self.op.apply_gradients(self.grads_and_vars)
        

        # initialize all variables in our graph
        init = tf.global_variables_initializer()
        agent.sess.run(init)

        # keep track of our reward for every episode
        self.average_rewards = tf.placeholder(dtype = tf.float32)

        # create our session graph in tensorboard
        self.writer.add_graph(agent.sess.graph)
        self.create_summaries()

        
    def get_action(self, state):
        """
        returns an action given the state (first part of our network)
        action is determined based on the max value of our outputs
        """
        state = np.reshape(state,[1,1])
        self.chosen_action = tf.argmax(self.outputs)
        feed_dict = {self.state: state}
        return agent.sess.run(self.chosen_action, feed_dict = feed_dict)


    def update_batch(self, states, actions, rewards):
        """ 
        updates our network by applying our gradients
        feeds in the entire batch to update on after many runs
        """
        rewards = np.concatenate(rewards)
        # make sure the values to be fed to the placeholders are of the write shape
        feed_dict = {self.state: states, self.action_holder: actions, self.reward_holder: rewards}
        agent.sess.run(self.update, feed_dict = feed_dict)


    def create_summaries(self):
        """ 
        creates all the summaries for tensorboard that we desire
        scalars are single values that can be plotted over time
        histograms are for multiple values and seeing their distributions
        """
        # keep track of our loss functions and average reward
        tf.summary.scalar("policy gradient loss", self.pg_loss)
        tf.summary.scalar("value loss", self.value_loss)
        tf.summary.scalar("total loss", self.total_loss)
        tf.summary.scalar("average reward", self.average_rewards)

        # keep track of our weights and biases for each layer
        #   tf.trainable_variables() has the weights then biases for every layer in the order they are created
        for i in range(self.hidden_layers):
            tf.summary.histogram("hidden"+str(i+1)+"_weights", tf.trainable_variables()[i*2])
            tf.summary.histogram("hidden"+str(i+1)+"_biases", tf.trainable_variables()[i*2+1])
        tf.summary.histogram("value weights", tf.trainable_variables()[-4])
        tf.summary.histogram("value biases", tf.trainable_variables()[-3])
        tf.summary.histogram("output weights", tf.trainable_variables()[-2])
        tf.summary.histogram("output bias", tf.trainable_variables()[-1])
        
        # merge all summaries into one summary, we can just run this summary to record everything
        self.merge = tf.summary.merge_all()


    def write_summaries(self, states, actions, rewards, average_rewards, i):
        """
        record the values that we created summaries out of, write them to tensorboard
        """
        # make sure the values to be fed to the placeholders are of the write shape
        feed_dict = {self.state: states, self.action_holder: actions,
                     self.reward_holder: rewards, self.average_rewards: average_rewards}
        summary = agent.sess.run(self.merge, feed_dict = feed_dict)
        self.writer.add_summary(summary, i)
        self.writer.flush()


    def update_state_values(self, states, rewards):
        """
        trains the state-value calculator
        """
        feed_dict = {self.state: states, self.reward_holder: rewards}
        agent.sess.run(self.value_update, feed_dict = feed_dict)
















"""
Policy-Gradient Agent

Graph takes in state (most likely timestep) and outputs the action as the max value

Calculates gradients for taken action through cross entropy
    Compiles all the gradients into a list before updating
    Calculates the mean gradients multiplied by the reward for that instance

Also writes the average reward, weights, and biases to tensorboard

Agent is initialized with adjustable hyper-parameters for the network

"""


class policy_gradient_agent(agent):

    def __init__(self, obs_space, action_space,
                 one_hot = True,                # whether or not to encode the state
                 hidden_layers = 1,             # the number of hidden layers
                 hidden_nodes = 20,             # the nodes per hidden layer
                 kernel_init = "variance_scaling",# the weights initializer for each layer
                 bias_init = "zeros",           # the bias initailizer for each layer
                 normalize = True,             # whether or not to normalize the rewards before updating
                 learning_rate = 1e-2,          # learning rate for optimizer
                 optimizer = "Adam",            # type of optimizer
                 directory = None):             # directory name for tensorboard

        """
        creates an agent with a network and training procedure
        """
        
        super().__init__(directory = directory)

        super().reset()
        
        kernel_init_dict = {"variance_scaling": tf.initializers.variance_scaling,
                            "random_normal": tf.initializers.random_normal,
                            "random_uniform": tf.initializers.random_uniform}
        bias_init_dict = {"zeros": tf.initializers.zeros,
                          "ones": tf.initializers.ones,
                          "variance_scaling": tf.initializers.variance_scaling}

        self.obs_space = obs_space # the number of states
        self.action_space = action_space # the number of actions (output)

        # set all our hyper-parameters
        self.hidden_layers = hidden_layers
        self.hidden_nodes = hidden_nodes
        self.kernel_init = kernel_init_dict[kernel_init]
        self.bias_init = bias_init_dict[bias_init]
        self.normalize = normalize
        self.lr = learning_rate

        # set our optimizer with given name and learning rate
        op_dict = {"Adam": tf.train.AdamOptimizer(self.lr),
                   "Gradient Descent": tf.train.GradientDescentOptimizer(self.lr)}
        self.op = op_dict[optimizer]

        # create our summary writer
        self.writer = tf.summary.FileWriter(self.directory)



        # ------------start building our network-------------
        
        #   name scopes are used for the tensorboard graph


        # our input consists of our state
        #   change to a one-hot format to make it easier for our network to recognize
        with tf.name_scope("network_inputs"):
            self.state = tf.placeholder(shape = [None,1],dtype = tf.int32)
            self.state_one_hot= tf.one_hot(self.state, obs_space)

        # keep track of our previous layer to feed to the next layer
        previous= self.state_one_hot

        # create our hidden layers using the new_hidden_layer method
        for i in range(hidden_layers):
            with tf.name_scope("hidden"+str(i+1)):
                new_hidden_layer = super().new_dense_layer(self.hidden_nodes,self.kernel_init,self.bias_init,previous)
                previous = new_hidden_layer

        with tf.name_scope("outputs"):
            self.outputs = super().new_dense_layer(self.action_space,self.kernel_init,self.bias_init,previous)
            self.outputs = tf.squeeze(self.outputs)


        # -----------set up our network's training procedure----------

        # inputs for our gradients and 'loss'
        #   only care about the action holder,
        #   reward holder is only for display purposes
        with tf.name_scope("back-prop_inputs"):
            self.action_holder = tf.placeholder(shape = [None], dtype = tf.int32)
            self.reward_holder = tf.placeholder(shape = [None], dtype = tf.float32)

        # calculate difference via cross entropy
        #   will act as our dependent variable to caluclate the gradients with
        with tf.name_scope("cross-entropy_and_loss"):
            self.cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits = self.outputs, labels = self.action_holder, name = "cross-entropy")

        # define our gradients and optimizers
        #   
        with tf.name_scope("optimizer"):
            self.grads = tf.gradients(self.cross_entropy, tf.trainable_variables())
            self.grad_placeholders = []
            for gradient in self.grads:
                grad_placeholder = tf.placeholder(shape = gradient.get_shape(), dtype = tf.float32)
                self.grad_placeholders.append(grad_placeholder)
            self.grads_and_vars = list(zip(self.grad_placeholders, tf.trainable_variables()))
            self.update = self.op.apply_gradients(self.grads_and_vars)
            self.grad_list = [] # updated when we run the network

        # initialize all variables in our graph
        init = tf.global_variables_initializer()
        agent.sess.run(init)

        # keep track of our reward for every episode
        self.average_rewards = tf.placeholder(dtype = tf.float32)

        # create our session graph in tensorboard
        self.writer.add_graph(agent.sess.graph)
        self.create_summaries()


    def get_action(self, state):
        """
        returns an action given the state (first part of our network)
        action is determined based on the max value of our outputs

        also gets the gradients based on the action taken
        """
        state = np.reshape(state,[1,1])
        self.chosen_action = tf.argmax(self.outputs)
        feed_dict = {self.state: state}
        action = agent.sess.run(self.chosen_action, feed_dict = feed_dict)

        # return our chosen action
        return action


    def update_batch(self, states, actions, rewards): # don't actually use the states and actions, there for OOP purposes
        """ 
        updates our network by applying our gradients multiplied by the rewards
        feed in the mean gradients to update our network
        """
        if self.normalize == True:
            rewards = self.normalize_rewards(rewards)

        # multiply gradients by normalized rewards
        shape = list(np.array(rewards).shape)
        shape.append(np.array(self.grad_list).shape[1])
        self.grad_list = np.reshape(self.grad_list, shape)
        feed_dict = {}
        for index, grad_placeholder in enumerate(self.grad_placeholders):
            mean_grads = np.mean([reward * self.grad_list[episode][step][index]
                for episode, episode_reward in enumerate(rewards)
                for step, reward in enumerate(episode_reward)],
                axis = 0)
            feed_dict[grad_placeholder] = mean_grads
        agent.sess.run(self.update, feed_dict = feed_dict)

        # reset our gradient list
        self.grad_list = []


    def create_summaries(self):
        """ 
        creates all the summaries for tensorboard that we desire
        scalars are single values that can be plotted over time
        histograms are for multiple values and seeing their distributions
        """
        # keep track of our average reward and 'loss'
        tf.summary.scalar("average reward", self.average_rewards)
        tf.summary.scalar("cross entropy", self.cross_entropy[0])

        # keep track of our weights and biases for each layer
        #   tf.trainable_variables() has the weights then biases for every layer in the order they are created
        for i in range(self.hidden_layers):
            tf.summary.histogram("hidden"+str(i+1)+"_weights", tf.trainable_variables()[i*2])
            tf.summary.histogram("hidden"+str(i+1)+"_biases", tf.trainable_variables()[i*2+1])
        tf.summary.histogram("output weights", tf.trainable_variables()[-2])
        tf.summary.histogram("output bias", tf.trainable_variables()[-1])
        
        # merge all summaries into one summary, we can just run this summary to record everything
        self.merge = tf.summary.merge_all()


    def write_summaries(self, states, actions, rewards, average_rewards, i):
        """
        record the values that we created summaries out of, write them to tensorboard
        """
        # make sure the values to be fed to the placeholders are of the write shape
        feed_dict = {self.state: states, self.action_holder: actions,
                     self.reward_holder: rewards, self.average_rewards: average_rewards}
        summary = agent.sess.run(self.merge, feed_dict = feed_dict)
        self.writer.add_summary(summary, i)
        self.writer.flush()


    def update_gradients(self, state, action):
        """
        calculate our gradients and append them to ongoing list
        """
        state = np.reshape(state,[1,1])
        feed_dict = {self.state: state, self.action_holder: [action]}
        gradients = agent.sess.run(self.grads, feed_dict = feed_dict)
        self.grad_list.append(gradients)

    def normalize_rewards(self, rewards):
        """
        Takes in a 2D array for rewards and normalizes it to be mean 0 and standard dev 1
        """
        flat_array = np.concatenate(rewards)
        array_mean = flat_array.mean()
        array_std = flat_array.std()
        return [(value - array_mean) / array_std
                for value in rewards]
        






