from gym.wrappers import Monitor


def evaluate(env, model, ep_num=100):
    env_test = env

    sum_reward = 0

    for _ in range(ep_num):

        obs = env_test.reset()
        score = 0
        done = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)

            obs, reward, done, _ = env_test.step(action)

            score += reward

        sum_reward += score

    env_test.close()

    return sum_reward / ep_num


def evaluate_agent(env, agent, ep_num=100):
    env_test = env

    sum_reward = 0

    for _ in range(ep_num):

        obs = env_test.reset()
        score = 0
        done = False

        while not done:
            action, _ = agent.act(obs)

            obs, reward, done, _ = env_test.step(action)

            score += reward

        sum_reward += score

    env_test.close()

    return sum_reward / ep_num


def create_videos(env, model, ep_num=2, folder="videos"):
    env = Monitor(env, folder, force=True, video_callable=lambda episode_id: True)

    sum_reward = 0

    for _ in range(ep_num):

        obs = env.reset()
        score = 0
        done = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)

            obs, reward, done, _ = env.step(action)

            score += reward

        sum_reward += score

    env.close()

    return sum_reward / ep_num
