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

        iterated_V = np.copy(V) #make copy of value map
        values_have_converged = False
        while delta > threshold:
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
            V = np.copy(iterated_V)
                
        return V

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
        policy[:, 1] = 1 # Initialise policy to choose second action systematically (for all states)
        V = np.zeros(env.get_state_size()) # Initialise value function to 0  
        T = env.get_T() #transition matrix
        absorbing_states = env.get_absorbing()
        R = env.get_R() #rewards matrix
        policy_stable = False # Condition to stop the main loop

        while not policy_stable:
            policy_stable = True
            
            V = self.evaluate_policy(env, policy, threshold=threshold)

            for curr_state in range(env.get_state_size()):
                
                if absorbing_states[0, curr_state] != 1:

                    prev_action = np.argmax(policy[curr_state,:]) #get index of action with max probability for s
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

        return policy, V

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
        policy, V = self.policy_iteration(env)

        return policy, V


# This class define the Monte-Carlo agent

class MC_agent(object):
    def __init__(self):
        self.epsilon = 0.4
        self.num_episodes = 10000
        self.environment = None
        self.policy = None

    # [Action required]
    # WARNING: make sure this function can be called by the auto-marking script
    def generate_action(self, state):

        action_probabilities = self.policy[state, :] #retrieve probabilities for each action at state s

        #return an action based on their probabilities
        return np.random.choice(np.arange(self.environment.get_action_size()), p = action_probabilities) 

    def calculate_returns(self,episode):
        """"""
        #add up rewards since states first occurence
        G = 0 
        returns = []
        reversed_episode = episode[::-1] #need to move backwards from terminal state, with its reward being 0 by definition

        for i, (state, action, reward) in enumerate(reversed_episode):
            if i != 0:
                returns.append((state, action, G))
            
            G = reward + self.environment.get_gamma()*G
            
        returns = np.array(returns[::-1]) #need to reverse list to have in order of state visited  

        return returns     

    def episode(self):
        """"""
        step, state, reward, done = self.environment.reset() #reset environment      
        episode_in_course = True


        episode = [] #initialise episode list

        #while episode hasn't finished continue 
        while episode_in_course:
            #choose best action
            best_action = self.generate_action(state)
            episode.append((state, best_action, reward))

            #make a step in the environment based on this action
            step, next_state, reward, done =  self.environment.step(best_action)

            #break if done is True
            if done: 
                episode.append((next_state, None, reward))
                episode_in_course = False

            state = next_state #if not done, update state variable to be the next state

        episode = np.array(episode)

        return episode


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
         self.environment = env
         V = np.zeros(env.get_state_size())
         Q = np.random.rand(env.get_state_size(), env.get_action_size()) 
         policy = np.zeros((env.get_state_size(), env.get_action_size())) 
         policy += self.epsilon/env.get_action_size() #initialise actions to have same probability based on epsilon
        
         #initialise epsilon greedy policy from random Q values
         for state in range(env.get_state_size()):
            best_action = np.argmax(Q[state])
            for action in range(env.get_action_size()):
                if action == best_action:
                    policy[state, action] += 1 - self.epsilon 

         values = [V]
         total_rewards = []


         total_occurences = np.zeros((env.get_state_size(), env.get_action_size()))
         returns = np.zeros((env.get_state_size(), env.get_action_size()))

         for i in range(1,self.num_episodes+1):
            print(i)
            self.policy = policy
            episode = self.episode() #generate an episode 

            #get state-action returns from episode
            every_visit_returns = self.calculate_returns(episode)
            unique_state_action_pairs = set([(state,action) for state, action in every_visit_returns[:,:2]])

            #iterate over all unique state action pairs in episode
            for (state, action) in unique_state_action_pairs:
                state, action = list(map(int, [state,action])) #make integers for indexing
                first_occurence = [i for i,(s,a) in enumerate(episode[:,:2]) if s == state and a == action][0] 
                G = every_visit_returns[first_occurence][2]

                #add G to total returns for state action pair
                returns[state,action] += G
                total_occurences[state, action] += 1

                #input average return for state, action pair as its value in Q
                Q[state, action] = returns[state, action] / total_occurences[state, action]

                best_action = np.argmax(Q[state, :]) #action is that with maximum state-action value
                
                for action in range(env.get_action_size()):
                    if action == best_action:
                        #make action with largest value have probability 1
                        policy[state, action] = 1 - self.epsilon + (self.epsilon/env.get_action_size())
                    else: 
                        policy[state, action] = (self.epsilon/env.get_action_size())

         for state in range(env.get_state_size()):
            #get value of each state (max Q(s,a))
            values[-1][state] = np.max(Q[state])

            #get total rewards for each state
            state_reward = 0
            for action in range(env.get_action_size()):
                state_reward += returns[state, action]

            total_rewards.append(state_reward)
            
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