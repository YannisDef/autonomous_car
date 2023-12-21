import gymnasium as gym
import numpy as np
from scipy import stats
from stable_baselines3 import PPO
from stable_baselines3 import A2C
from stable_baselines3 import DQN
from statsmodels.stats.proportion import proportions_ztest

def analyse(model, env):
    """ Function to analyse your model during 100 episodes.

    Args:
        model: model you want to analyse
        env: envionnement () to test your model

    Returns:
        tuple: tuple(list_reward, list_speed, list_accident)
    """
    liste_reward= []
    liste_speed = []
    liste_accident = []
    for _ in range(100):
        done = truncated = False
        obs, _ = env.reset()
        sum_rew = 0
        sum_speed = 0
        time_step = 0
        while not (done or truncated):
           action, _ = model.predict(obs, deterministic=True)
           obs, reward, done, truncated, _ = env.step(action)

           sum_speed += obs[0][3]
           sum_rew += reward
           time_step += 1
           env.render()
        liste_accident.append(1 if time_step < 30 else 0)
        liste_speed.append(20*sum_speed/time_step)
        liste_reward.append(sum_rew)
    return np.array(liste_reward), np.array(liste_speed), np.array(liste_accident)

# Load saved model
#! Be careful with your path
model = DQN.load("graphique_l2\DQN_highway_fast_env_long_50000")
model2 = A2C.load("graphique_l2\A2C_highway_fast_env_long_50000")
model3 = PPO.load("graphique_l2\PPO_highway_fast_env_long_50000")

# Create an env
env = gym.make("highway-fast-v0", render_mode="human")
env.config['normalize_reward'] = False
env.config['vehicles_density'] = 1

r1, s1, a1 = analyse(model, env)
r2, s2, a2 = analyse(model2, env)
r3, s3, a3 = analyse(model3, env)

print(np.mean(r1))
print(np.mean(r2))
print(np.mean(r3))

print(np.mean(s1))
print(np.mean(s2))
print(np.mean(s3))

print(np.mean(a1))
print(np.mean(a2))
print(np.mean(a3))

# Réalisation du test ANOVA
f_statistic, p_value = stats.f_oneway(r1, s1, a1)

print("F-statistique:", f_statistic)
print("P-valeur:", p_value)

f_statistic, p_value = stats.f_oneway(r2, s2, a2)

print("F-statistique:", f_statistic)
print("P-valeur:", p_value)

table = np.array([[21, 79], [8, 92]])
res = chi2_contingency(table)
print(res)
