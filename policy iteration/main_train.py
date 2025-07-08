from environment import grid_world
from agent import AGENT


WORLD_HEIGHT = 5
WORLD_WIDTH = 10

env = grid_world(WORLD_HEIGHT,WORLD_WIDTH,
                 GOAL = [[0,0], [WORLD_HEIGHT-1, WORLD_WIDTH-1]],
                 OBSTACLES= [[2,5], [1,2], [2,2], [3,2], [2, 8],[3, 8],[4, 8]])
agent = AGENT(env,is_upload=False)
agent.policy_iteration()