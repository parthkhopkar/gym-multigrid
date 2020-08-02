import gym
import time
from gym.envs.registration import register
from gym_multigrid.envs import *
import argparse

parser = argparse.ArgumentParser(description=None)
parser.add_argument('-e', '--env', default='empty', type=str)

args = parser.parse_args()

def main():

    if args.env == 'soccer':
        register(
            id='multigrid-soccer-v0',
            entry_point='gym_multigrid.envs:SoccerGame4HEnv10x15N2',
        )
        env = gym.make('multigrid-soccer-v0')

    elif args.env == 'collect_game':
        register(
            id='multigrid-collect-v0',
            entry_point='gym_multigrid.envs:CollectGame4HEnv10x10N2',
        )
        env = gym.make('multigrid-collect-v0')

    else:
        env = gym.make('multigrid-empty-v0')


    _ = env.reset()

    nb_agents = len(env.agents)

    while True:
        env.render(mode='human', highlight=True)
        time.sleep(0.1)

        ac = [env.action_space.sample() for _ in range(nb_agents)]
        obs, rewards, done, _ = env.step(ac)
        print(obs[0].shape)
        if done:
            break

if __name__ == "__main__":
    main()