import numpy as np 

# This class define the Dynamic Programing agent 
class DP_agent(object):

    def evaluate_policy(self, env, policy, threshold = 0.0001):
        """
        Policy evaluation on GridWorld
        input: 
        - policy {np.array} -- policy to evaluate
        - threshold {float} -- threshold value used to stop the policy evaluation algorithm
        - gamma {float} -- discount factor
        output: 
        - V {np.array} -- value function corresponding to the policy 
        - epochs {int} -- number of epochs to find this value function
        """
        
        # Ensure inputs are valid
        assert (policy.shape[0] == env.get_state_size()) and (policy.shape[1] == env.get_action_size()), "The dimensions of the policy are not valid."
        assert (env.get_gamma() <=1) and (env.get_gamma() >= 0), "Discount factor should be in [0, 1]."

        # Initialisation
        delta = 2*threshold # Ensure delta is bigger than the threshold to start the loop
        V = np.zeros(env.get_state_size()) # Initialise value function to 0  
        T = env.get_T() #transition matrix
        absorbing_states = env.get_absorbing()
        R = env.get_R() #rewards matrix
        epoch = 0

        iterated_V = np.copy(V) #make copy of value map
        values_have_converged = False
        while not values_have_converged:
            epoch += 1
            for curr_state in range(env.get_state_size()):
                #check if state is absorbing --> continue if it is
                if absorbing_states[0, curr_state] != 1:
                    V_s_list = [] #initialise state value list prior to looping over all actions
                    for action in range(env.get_action_size()):
                        Q_list = [] #initialise state action value list 
                        for next_state in range(env.get_state_size()):
                            Q = (T[curr_state, next_state, action]*(R[curr_state, next_state, action] 
                                     + env.get_gamma()*V[next_state])
                                    )       

                            Q_list.append(Q)

                        #calculate value of current state from subsequent state values and the probability 
                        #of the policy of ocurring
                        Q_s_a = sum(Q_list)
                        V_s_list.append(policy[curr_state, action]*Q_s_a)
                    
                    V_s = sum(V_s_list)
                    #update value of current state    
                    iterated_V[curr_state] = V_s

            #calculate delta for current iteration
            delta = max(abs(iterated_V - V))

            #update value map to one retrieved from iteration
            V = iterated_V

            #check for threshold
            if delta < threshold: 
                values_have_converged = True
                
        return V, epoch

    def policy_iteration(self, env, threshold = 0.0001):
        """
        Policy iteration on GridWorld
        input: 
        - threshold {float} -- threshold value used to stop the policy iteration algorithm
        - gamma {float} -- discount factor
        output:
        - policy {np.array} -- policy found using the policy iteration algorithm
        - V {np.array} -- value function corresponding to the policy 
        - epochs {int} -- number of epochs to find this policy
        """

        # Ensure gamma value is valid
        assert (env.get_gamma() <=1) and (env.get_gamma() >= 0), "Discount factor should be in [0, 1]."

        # Initialisation
        policy = np.zeros((env.get_state_size(), env.get_action_size())) # Vector of 0
        policy[:, 0] = 1 # Initialise policy to choose action 1 systematically (for all states)
        V = np.zeros(env.get_state_size()) # Initialise value function to 0  
        T = env.get_T() #transition matrix
        absorbing_states = env.get_absorbing()
        R = env.get_R() #rewards matrix
        total_epochs = 0
        policy_stable = False # Condition to stop the main loop

        while not policy_stable:
            policy_stable = True
            
            V, epochs = self.evaluate_policy(env, policy, threshold=threshold)
            total_epochs += epochs

            for curr_state in range(env.get_state_size()):
                
                if absorbing_states[0, curr_state] != 1:
                    curr_state_policies = policy[curr_state] #retrieve probabilities of actions for current state
                    prev_action = curr_state_policies.index(max(policy[curr_state])) #get index of action with max probability

                    Q_new = [] #initialise state_action function for each action
                    #iterate over all actions and calculate new value function 
                    for action in range(env.get_action_size()):
                        Q_s_a = 0
                        for next_state in range(env.get_state_size()):
                            Q_s_a += (T[curr_state, next_state, action]*(R[curr_state, next_state, action] 
                                     + env.get_gamma()*V[next_state])
                                    )

                        Q_new.append(Q_s_a) #append value of action to list of new values

                #get new policy
                updated_policy = np.zeros(env.get_action_size())
                updated_policy[np.argmax(Q_new)] = 1 #make action with largest value have probability 1
                policy[curr_state,:] = updated_policy #update old policy

                #keep iterating until stable policy is reached
                if prev_action != np.argmax(updated_policy):
                    policy_stable = False

        return policy, V, total_epochs

    # [Action required]
    # WARNING: make sure this function can be called by the auto-marking script
    def solve(self, env):
        """
        Solve a given Maze environment using Dynamic Programming (policy iteration).
        input: env {Maze object} -- Maze to solve
        output: 
        - policy {np.array} -- Optimal policy found to solve the given Maze environment 
        - V {np.array} -- Corresponding value function 
        """
        #use policy iteration to arrive to optimum policy in maze
        policy, V, epochs = self.policy_iteration(env)

        return policy, V


# This class define the Monte-Carlo agent

class MC_agent(object):
  
  # [Action required]
  # WARNING: make sure this function can be called by the auto-marking script
  def solve(self, env):
    """
    Solve a given Maze environment using Monte Carlo learning
    input: env {Maze object} -- Maze to solve
    output: 
      - policy {np.array} -- Optimal policy found to solve the given Maze environment 
      - values {list of np.array} -- List of successive value functions for each episode 
      - total_rewards {list of float} -- Corresponding list of successive total non-discounted sum of reward for each episode 
    """

    # Initialisation (can be edited)
    Q = np.random.rand(env.get_state_size(), env.get_action_size()) 
    V = np.zeros(env.get_state_size())
    policy = np.zeros((env.get_state_size(), env.get_action_size())) 
    values = [V]
    total_rewards = []

    #### 
    # Add your code here
    # WARNING: this agent only has access to env.reset() and env.step()
    # You should not use env.get_T(), env.get_R() or env.get_absorbing() to compute any value
    ####
    
    return policy, values, total_rewards

# This class define the Temporal-Difference agent

class TD_agent(object):

  # [Action required]
  # WARNING: make sure this function can be called by the auto-marking script
  def solve(self, env):
    """
    Solve a given Maze environment using Temporal Difference learning
    input: env {Maze object} -- Maze to solve
    output: 
      - policy {np.array} -- Optimal policy found to solve the given Maze environment 
      - values {list of np.array} -- List of successive value functions for each episode 
      - total_rewards {list of float} -- Corresponding list of successive total non-discounted sum of reward for each episode 
    """

    # Initialisation (can be edited)
    Q = np.random.rand(env.get_state_size(), env.get_action_size()) 
    V = np.zeros(env.get_state_size())
    policy = np.zeros((env.get_state_size(), env.get_action_size())) 
    values = [V]
    total_rewards = []

    #### 
    # Add your code here
    # WARNING: this agent only has access to env.reset() and env.step()
    # You should not use env.get_T(), env.get_R() or env.get_absorbing() to compute any value
    ####
    
    return policy, values, total_rewards