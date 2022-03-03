import numpy as np
from numpy.core.fromnumeric import shape
from numpy.random import randint

class GridWorld():
    def __init__(self, wraparound = True , randstart = True) -> None:
        ''' Intialize the windy grid world enviroment 
        hardcoded dimension of 500 rows by 100 col
        hardcoded reward location

        @param wraparound: Boolean var, indicates whether reaching top of the grid results in
            wrap around or end of episode
        @param randstart: Booelean var, indicates whether start point is random col of bottom row 
            or middle of bottom row
        '''
        self.rows = 20
        self.col = 10
        # Reward location matrix, first col of each row indicates if reward is taken or not
        # 2nd and 3rd column indicate postion of reward
        self.rewards_loc = np.array([[1,0,self.col//2]])
        # rewards_loc formatted as matrix 
        # [[amount,loc_x,loc_y],
        # [amount,loc_x,loc_y]]

        # Remaining rewards
        self.rewards = np.sum(self.rewards_loc[:,0])              
        self.wrap = wraparound
        self.rand = randstart
        self.discount = 0.98
        if self.rand:
            self.agent_location = [self.rows-1,np.random.randint(self.col)]
        else:
            self.agent_location = [self.rows-1,(self.col-1)//2]

    def take_action(self, action) -> int:
        
        # Copy current state
        self.prev_location = [self.agent_location[0],self.agent_location[1]]
        # take action 0:stay, 1: left, 2:right
        if action == 1:
            self.agent_location[1] -= 1
        elif action == 2:
            self.agent_location[1] += 1
        
        # move up
        self.agent_location[0] -= 1

        # Get number of rewards on map
        rew = self.rewards_loc.shape[0]
        
        # Reward of taking the action
        action_rew = 0

        # Check if we hit a reward
        i = 0 
        while i<rew:
            # Check if rew is not claimed
            if self.rewards_loc[i,0] == 1:
                # Check if agent is at reward
                if self.agent_location[0] == self.rewards_loc[i,1] and self.agent_location[1] == self.rewards_loc[i,2]:
                    # Assign reward to the agent
                    action_rew = 1
                    # Remove reward from the gridworld
                    self.rewards_loc[i,0] = 0
                    # Reduce total rewards by 1
                    self.rewards -= 1
                    
                    # Check if no rewards remain
                    if self.rewards == 0:
                        # End of episode 
                        self.discount = 0

            i += 1

        # Check side boundry
        if self.agent_location[1] == -1 or self.agent_location[1] == self.col:
            self.discount = 0
            self.agent_location = [self.rows-1,(self.col-1)//2]
        
        # Chech upper boundry
        if self.agent_location[0] == -1:
            if self.wrap is True:
                self.agent_location[0] = self.rows-1
            else:
                self.discount = 0
                self.agent_location = [self.rows-1,(self.col-1)//2]


        return action_rew

    def reset(self):

        self.rewards_loc = np.array([[1,0,self.col//2]])
        self.rewards = np.sum(self.rewards_loc[:,0])

        self.discount = 0.98

        if self.rand:
            self.agent_location = [self.rows-1,np.random.randint(self.col)]
        else:
            self.agent_location = [self.rows-1,(self.col-1)//2]
    
    def get_state(self):
        
        return [self.agent_location[0],self.agent_location[1]]

    def get_prev_state(self):
        return self.prev_location

    def get_disct(self):
        return self.discount



    










