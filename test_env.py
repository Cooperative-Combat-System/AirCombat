import numpy as np
from envs.JSBSim.envs import SingleCombatEnv, SingleControlEnv, MultipleCombatEnv,AirCombatEnv,AirCombatEnvV1
from envs.env_wrappers import SubprocVecEnv, ShareDummyVecEnv, ShareSubprocVecEnv, DummyVecEnv
from envs.JSBSim.core.catalog import Catalog as c
import logging
import time
logging.basicConfig(level=logging.DEBUG)

def test_env():
    parallel_num = 1
    envs = DummyVecEnv([lambda: AirCombatEnvV1("1v1/MyConfig/myaircombat_v1") for _ in range(parallel_num)])

    envs.reset()
    # DataType test
    obs_shape = (parallel_num, envs.num_agents, *envs.observation_space.shape)
    # act_shape = (parallel_num, envs.num_agents, *envs.action_space.shape)
    reward_shape = (parallel_num, envs.num_agents, 1)
    done_shape = (parallel_num, envs.num_agents, 1)


    def convert(sample):
        return np.concatenate((sample[0], np.expand_dims(sample[1], axis=0)))

    episode_reward = 0
    step = 0
    print("testing")
    while True:
        actions = np.array([[envs.action_space.sample() for _ in range(envs.num_agents)] for _ in range(parallel_num)])
        obss, rewards, dones, infos = envs.step(actions)
        # bloods = [envs.envs[0].agents[agent_id].bloods for agent_id in envs.envs[0].agents.keys()]
        print(f"step:{step}")
        episode_reward += rewards[:,0,:]
        # envs.render(mode='txt', filepath='JSBSimRecording.txt.acmi')
        # terminate if any of the parallel envs has been done
        if np.all(dones):
            print(episode_reward)
            break
        step += 1
    envs.close()

def test_env_v1(num_episodes=5, max_steps=1000):
    parallel_num = 1  # 需要并行就改 >1
    envs = DummyVecEnv([lambda: AirCombatEnvV1("1v1/MyConfig/myaircombat_v1")
                        for _ in range(parallel_num)])

    print("testing")

    for ep in range(num_episodes):
        print(f"Episode {ep + 1}")
        _ = envs.reset()
        step = 0
        # 累计奖励：形状 [n_env, n_agent]
        ep_return = np.zeros((parallel_num, envs.num_agents), dtype=np.float32)

        while True:
            # 随机动作（测试用）：形状 [n_env, n_agent, act_dim]
            actions = np.array([
                [envs.action_space.sample() for _ in range(envs.num_agents)]
                for _ in range(parallel_num)
            ])

            obss, rewards, dones, infos = envs.step(actions)
            # rewards: [n_env, n_agent, 1] -> squeeze 到 [n_env, n_agent]
            ep_return += np.squeeze(rewards, axis=-1)

            # 判定本回合是否结束
            # dones: [n_env, n_agent, 1]，若“本 env 的所有 agent 都 done”则该 env 结束
            done_env = np.all(dones, axis=(1, 2))  # 形状 [n_env]

            step += 1
            if np.all(done_env) or step >= max_steps:
                # 结束当前 episode
                print(f"[Episode {ep+1}/{num_episodes}] steps={step}")
                print("  return per env × agent:\n", ep_return)
                break

        # 如果你希望每个 episode 后把轨迹写文件/渲染，可以在这里做
        # envs.render(mode='txt', filepath=f'episode_{ep}.acmi')

    envs.close()


def test_multi_env():
    parallel_num = 1
    envs = ShareDummyVecEnv([lambda: MultipleCombatEnv('2v2/NoWeapon/HierarchySelfplay') for _ in range(parallel_num)])
    assert envs.num_agents == 4
    obs_shape = (parallel_num, envs.num_agents, *envs.observation_space.shape)
    share_obs_shape = (parallel_num, envs.num_agents, *envs.share_observation_space.shape)
    reward_shape = (parallel_num, envs.num_agents, 1)
    done_shape = (parallel_num, envs.num_agents, 1)

    # DataType test
    obs, share_obs = envs.reset()
    step = 0
    envs.render(mode='txt', filepath='JSBSimRecording.txt.acmi')
    assert obs.shape == obs_shape and share_obs.shape == share_obs_shape
    while True:
        actions = np.array([[envs.action_space.sample() for _ in range(envs.num_agents)] for _ in range(parallel_num)])
        start = time.time()
        obs, share_obs, rewards, dones, info = envs.step(actions)
        bloods = [envs.envs[0].agents[agent_id].bloods for agent_id in envs.envs[0].agents.keys()]
        print(f"step:{step}, bloods:{bloods}")
        end = time.time()
        # print(rewards)
        envs.render(mode='txt', filepath='JSBSimRecording.txt.acmi')
        assert obs.shape == obs_shape and rewards.shape == reward_shape and dones.shape == done_shape and share_obs_shape
        if np.all(dones):
            break
        step += 1

    envs.close()

test_env()
