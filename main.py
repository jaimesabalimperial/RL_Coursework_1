from coursework1 import MC_agent, DP_agent, TD_agent
from maze import Maze

if __name__ == '__main__':
    # Example main (can be edited)

    ### Question 0: Defining the environment
    print("Creating the Maze:\n")
    maze = Maze()

    ### Question 1: Dynamic programming
    #dp_agent = DP_agent()
    #dp_policy, dp_value = dp_agent.solve(maze)

    #print("Results of the DP agent:\n")
    #maze.get_graphics().draw_policy(dp_policy)
    #maze.get_graphics().draw_value(dp_value)

    ### Question 2: Monte-Carlo learning
    mc_agent = MC_agent()
    mc_policy, mc_values, total_rewards = mc_agent.solve(maze)

    print("Results of the MC agent:\n")
    maze.get_graphics().draw_policy(mc_policy)
    maze.get_graphics().draw_value(mc_values[-1])

    ### Question 3: Temporal-Difference learning
    #td_agent = TD_agent()
    #td_policy, td_values, total_rewards = td_agent.solve(maze)

    #print("Results of the TD agent:\n")
    #maze.get_graphics().draw_policy(td_policy)
    #maze.get_graphics().draw_value(td_values[-1])