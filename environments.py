import numpy as np



class environment:
    """
    Environment super-class
    Represents environment for the agent

    The environment must be able to be represented as states
    There should be a set of possible actions to take on the environment
    It should return rewards when we transition between timesteps

    """

    def __init__(self, data):
        """
        initializes the environment with some data that can represent it
        """
        self.data = data

    def get_data(self):
        """
        returns the data that primarily represents this environment
        """
        return self.data

    def get_obs_space(self):
        raise NotImplementedError

    def get_action_space(self):
        raise NotImplementedError
    
    def take_action(self):
        raise NotImplementedError

    def get_first_state(self):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

        



class time_environment(environment): # might need to think of a better name

    """
    The data-set should be a 2D array:
    row # represents the timestep
    column # represents the action
    number in [i,j] is the value of choosing action j at timestep i

    e.g.:
    would represent a 3-timestep progression with 4 possible actions
    data = [[ _, _, _, _],
            [ _, _, _, _],
            [ _, _, _, _]]
    """

    def __init__(self, data, final_reward_subtraction = 3):
        super().__init__(data) # data needs to be a 2D matrix
        self.final_subtract = final_reward_subtraction # how much to subtract for missing an object
        self.action_space = len(data[0]) # action space = number of objects
        self.obs_space = len(data) # observation space = number of timesteps
        self.timestep = 0 # the timestep, aka our state
        self.collection = np.zeros(shape = [self.action_space], dtype = np.int32)

    def take_action(self, action):
        """
        updates our state (not the actual network input, just what actions we have aquired to this point)
        returns a reward based on the number in the 'data' matrix
            also returns a final reward based on diversity of objects

        returns our state (the timestep), the reward, and whether or not the episode is over
        """

        # get the reward for the value in the 'data' matrix
        # we're going to call  this VALUE REWARD
        value_reward = super().get_data()[self.timestep][action]
        diversity_reward = 0

        # update our list of what objects we do and don't have as of now in this episode
        if self.collection[action]==0:
            self.collection[action] += 1

        # check if we are on the last timestep
        if self.timestep+1 == self.obs_space:
            done = True
            self.timestep = 0
            
            # calculate our final reward based on diversity only when the episode ends
            # we're going to call this DIVERSITY REWARD
            for val in self.collection:
                if val == 0:
                    diversity_reward -= self.final_subtract # deduct a certain value for every object not ever chosen
            self.collection = np.zeros(shape = [self.action_space], dtype = np.int32)

        else:
            done = False
            self.timestep += 1
        
        reward = value_reward + diversity_reward
        
        return self.timestep, reward, done


    def get_obs_space(self):
        """
        returns the number of states we can see, aka number of timesteps
        """
        return self.obs_space

    def get_action_space(self):
        """
        represents the number of actions we can take, aka number of objects we observe
        """
        return self.action_space

    def get_first_state(self):
        """
        returns the first timestep (0)
        """
        return 0

    def reset(self):
        """
        resets the timestep to 0
        """
        self.timestep = 0



    
