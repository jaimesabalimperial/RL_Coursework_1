from coursework1 import MC_agent, DP_agent, TD_agent
from maze import Maze
import matplotlib.pyplot as plt
import numpy as np

"""     plt.figure()
        plt.grid()
        for state in list(differences.keys()):
            episodes = np.arange(len(differences[state]))
            plt.plot(episodes, differences[state])

        plt.xlabel("Episode")
        plt.ylabel("Difference in State Value Between Episodes")
        plt.show()"""

def print_results(maze, MC=False, DP=False, TD= False, ):

    if MC == True:
        ### Question 2: Monte-Carlo learning
        mc_agent = MC_agent()
        mc_policy, mc_values, total_rewards = mc_agent.solve(maze)

        print("Results of the MC agent:\n")
        maze.get_graphics().draw_policy(mc_policy)
        maze.get_graphics().draw_value(mc_values[-1])

        episodes = np.arange(len(total_rewards))

        plt.figure()
        plt.grid()
        plt.plot(np.unique(episodes), np.poly1d(np.polyfit(episodes, total_rewards, 40))(np.unique(episodes)), "r")
        plt.xlabel("Episode")
        plt.ylabel("Total Non-Discounted Reward Retrieved from Episode")
        plt.show()

    elif DP == True:
        ### Question 1: Dynamic programming
        dp_agent = DP_agent()
        dp_policy, dp_value = dp_agent.solve(maze)

        print("Results of the DP agent:\n")
        maze.get_graphics().draw_policy(dp_policy)
        maze.get_graphics().draw_value(dp_value)


    elif TD == True:
        ### Question 3: Temporal-Difference learning
        td_agent = TD_agent()
        td_policy, td_values, total_rewards = td_agent.solve(maze)

        print("Results of the TD agent:\n")
        maze.get_graphics().draw_policy(td_policy)
        maze.get_graphics().draw_value(td_values[-1])

        episodes = np.arange(len(total_rewards))

        plt.figure()
        plt.grid()
        plt.plot(np.unique(episodes), np.poly1d(np.polyfit(episodes, total_rewards, 40))(np.unique(episodes)), "r")
        plt.xlabel("Episode")
        plt.ylabel("Total Non-Discounted Reward Retrieved from Episode")
        plt.show()

    else: 
        return None


if __name__ == '__main__':
    # Example main (can be edited)

    ### Question 0: Defining the environment
    print("Creating the Maze:\n")
    maze = Maze()

    print_results(maze, TD = True)

