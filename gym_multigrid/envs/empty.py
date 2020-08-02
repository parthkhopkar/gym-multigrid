from gym_multigrid.multigrid import *
from gym.envs.registration import register

class EmptyEnv(MultiGridEnv):
    """
    Empty grid environment, no obstacles, sparse reward
    """

    def __init__(
        self,
        size=5,
        agents_index=[],
        agent_start_pos=(1,1),
        agent_start_dir=0,
        view_size=5
    ):
        self.world = SmallWorld

        agents = []

        for i in agents_index:
            agents.append(Agent(self.world, i, view_size=view_size))
        
        # self.agent_start_pos = agent_start_pos
        # self.agent_start_dir = agent_start_dir

        super().__init__(
            grid_size=size,
            # max_steps=4*size*size,
            max_steps=128,
            # Set this to True for maximum speed
            see_through_walls=True,
            agents=agents,
            partial_obs=True,
            agent_view_size=view_size,
            actions_set=SmallActions,
            objects_set=self.world
        )

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        # self.grid.wall_rect(0, 0, width, height)
        self.grid.horz_wall(self.world, 0, 0)
        self.grid.horz_wall(self.world, 0, height-1)
        self.grid.vert_wall(self.world, 0, 0)
        self.grid.vert_wall(self.world, width-1, 0)

        # Place a goal square in the bottom-right corner
        self.put_obj(Goal(self.world, 1), width - 2, height - 2)
        
        # Place goal at centre
        # self.put_obj(Goal(self.world, 1), int((width - 1)/2), int((height -1)/2))
        
        # Place a goal square in the bottom-left corner
        # self.put_obj(Goal(self.world, 1), 1, height - 2)

        # Place the agents
        init_pos = (1,1)
        for agent in self.agents:
            # self.place_agent(agent)
            self.put_obj(agent, *init_pos)
            agent.pos = init_pos
            agent.init_pos = init_pos
            agent.dir = 0
            agent.init_dir = 0
            # agent.init_pos = (1,1)
            # agent.init_dir = 0
            # agent.dir = 0
        # if self.agent_start_pos is not None:
        #     self.agent_pos = self.agent_start_pos
        #     self.agent_dir = self.agent_start_dir
        # else:
        #     self.place_agent()

        # self.mission = "get to the green goal square"
    
    def _handle_occupy_goal(self, i, rewards, fwd_pos, fwd_cell):
        self.agents[i].terminated = True  # Terminate agent if it has reached goal state
        fwd_cell.toggle()
        self.grid.set(*self.agents[i].pos, None)
        self.agents[i].pos = fwd_pos
        rewards[i] = self._reward(i, rewards, 1)

class EmptyEnv5x5(EmptyEnv):
    def __init__(self, **kwargs):
        super().__init__(size=7,  # size + 2 because of the wall
        agents_index=[1],
        **kwargs)


register(
            id='multigrid-empty-v0',
            entry_point='gym_multigrid.envs:EmptyEnv5x5',
        )

# class EmptyRandomEnv5x5(EmptyEnv):
#     def __init__(self):
#         super().__init__(size=5, agent_start_pos=None)

# class EmptyEnv6x6(EmptyEnv):
#     def __init__(self, **kwargs):
#         super().__init__(size=6, **kwargs)

# class EmptyRandomEnv6x6(EmptyEnv):
#     def __init__(self):
#         super().__init__(size=6, agent_start_pos=None)

# class EmptyEnv16x16(EmptyEnv):
#     def __init__(self, **kwargs):
#         super().__init__(size=16, **kwargs)

# register(
#     id='MiniGrid-Empty-5x5-v0',
#     entry_point='gym_minigrid.envs:EmptyEnv5x5'
# )

# register(
#     id='MiniGrid-Empty-Random-5x5-v0',
#     entry_point='gym_minigrid.envs:EmptyRandomEnv5x5'
# )

# register(
#     id='MiniGrid-Empty-6x6-v0',
#     entry_point='gym_minigrid.envs:EmptyEnv6x6'
# )

# register(
#     id='MiniGrid-Empty-Random-6x6-v0',
#     entry_point='gym_minigrid.envs:EmptyRandomEnv6x6'
# )

# register(
#     id='MiniGrid-Empty-8x8-v0',
#     entry_point='gym_minigrid.envs:EmptyEnv'
# )

# register(
#     id='MiniGrid-Empty-16x16-v0',
#     entry_point='gym_minigrid.envs:EmptyEnv16x16'
# )