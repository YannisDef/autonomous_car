import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3 import A2C
from stable_baselines3 import DQN

# Environnent graphique 1
env = gym.make("highway-fast-v0")
env.config['normalize_reward'] = False
env.config['high_speed_reward'] = 0.4
env.config['collision_reward'] = -1
env.config['reward_speed_range'] = [20, 30]
env.config['right_lane_reward'] = 0.1
env.config['lane_change_reward'] = 0

# Environnent graphique 2
env_long = gym.make("highway-fast-v0")
env_long.config['normalize_reward'] = False
env_long.config['high_speed_reward'] = 0.4
env_long.config['collision_reward'] = -1
env_long.config['reward_speed_range'] = [20, 30]
env_long.config['right_lane_reward'] = 0.1
env_long.config['lane_change_reward'] = 0
env_long.config['duration'] = 200

# Environnent graphique 3 
env_long_d2 = gym.make("highway-fast-v0")
env_long_d2.config['vehicles_density'] = 2
env_long_d2.config['normalize_reward'] = False
env_long_d2.config['high_speed_reward'] = 0.4
env_long_d2.config['collision_reward'] = -1
env_long_d2.config['reward_speed_range'] = [20, 30]
env_long_d2.config['right_lane_reward'] = 0.1
env_long_d2.config['lane_change_reward'] = 0
env_long_d2.config['duration'] = 200

env_long_d3 = gym.make("highway-fast-v0")
env_long_d3.config['vehicles_density'] = 3
env_long_d3.config['normalize_reward'] = False
env_long_d3.config['high_speed_reward'] = 0.4
env_long_d3.config['collision_reward'] = -1
env_long_d3.config['reward_speed_range'] = [20, 30]
env_long_d3.config['right_lane_reward'] = 0.1
env_long_d3.config['lane_change_reward'] = 0
env_long_d3.config['duration'] = 200

env_long_d4 = gym.make("highway-fast-v0")
env_long_d4.config['vehicles_density'] = 4
env_long_d4.config['normalize_reward'] = False
env_long_d4.config['high_speed_reward'] = 0.4
env_long_d4.config['collision_reward'] = -1
env_long_d4.config['reward_speed_range'] = [20, 30]
env_long_d4.config['right_lane_reward'] = 0.1
env_long_d4.config['lane_change_reward'] = 0
env_long_d4.config['duration'] = 200

env_long_d5 = gym.make("highway-fast-v0")
env_long_d5.config['vehicles_density'] = 5
env_long_d5.config['normalize_reward'] = False
env_long_d5.config['high_speed_reward'] = 0.4
env_long_d5.config['collision_reward'] = -1
env_long_d5.config['reward_speed_range'] = [20, 30]
env_long_d5.config['right_lane_reward'] = 0.1
env_long_d5.config['lane_change_reward'] = 0
env_long_d5.config['duration'] = 200

# Environnent graphique 4 
env_long_d3_speed = gym.make("highway-fast-v0")
env_long_d3_speed.config['vehicles_density'] = 3
env_long_d3_speed.config['normalize_reward'] = False
env_long_d3_speed.config['high_speed_reward'] = 0.4
env_long_d3_speed.config['collision_reward'] = -1
env_long_d3_speed.config['reward_speed_range'] = [20, 30]
env_long_d3_speed.config['right_lane_reward'] = 0.1
env_long_d3_speed.config['lane_change_reward'] = 0
env_long_d3_speed.config['duration'] = 200
env_long_d3_speed.config['reward_speed_range'] = [10,20]

env_long_d3_collision = gym.make("highway-fast-v0")
env_long_d3_collision.config['vehicles_density'] = 3
env_long_d3_collision.config['normalize_reward'] = False
env_long_d3_collision.config['high_speed_reward'] = 0.4
env_long_d3_collision.config['collision_reward'] = -1
env_long_d3_collision.config['reward_speed_range'] = [20, 30]
env_long_d3_collision.config['right_lane_reward'] = 0.1
env_long_d3_collision.config['lane_change_reward'] = 0
env_long_d3_collision.config['duration'] = 200
env_long_d3_collision.config['collision_reward'] = -10

env_long_d3_lane = gym.make("highway-fast-v0")
env_long_d3_lane.config['vehicles_density'] = 3
env_long_d3_lane.config['normalize_reward'] = False
env_long_d3_lane.config['high_speed_reward'] = 0.4
env_long_d3_lane.config['collision_reward'] = -1
env_long_d3_lane.config['reward_speed_range'] = [20, 30]
env_long_d3_lane.config['right_lane_reward'] = 0.1
env_long_d3_lane.config['lane_change_reward'] = 0
env_long_d3_lane.config['duration'] = 200
env_long_d3_lane.config['lane_change_reward'] = 0.2

def training(env, env_name, model_name, dossier, number_it):
    if model_name=="A2C":
        model_A2C = A2C("MlpPolicy", env, verbose=1,seed=1, tensorboard_log = dossier+"/"+env_name+"_A2C_"+str(number_it)+"/")
        model_A2C.learn(int(number_it))
        model_A2C.save(dossier+"/"+model_name+"_"+env_name+"_"+str(number_it))
        
    if model_name=="DQN":
        model_DQN = DQN('MlpPolicy', env,
                    policy_kwargs=dict(net_arch=[256, 256]), 
                    learning_rate=5e-4,
                    buffer_size=15000,
                    learning_starts=200,
                    batch_size=32,
                    gamma=0.99,
                    train_freq=1,
                    gradient_steps=1,
                    target_update_interval=50,
                    verbose=1,
                    tensorboard_log = dossier+"/"+env_name+"_DQN_"+str(number_it)+"/"
        )
        model_DQN.learn(int(number_it))
        model_DQN.save(dossier+"/"+model_name+"_"+env_name+"_"+str(number_it))
        
    if model_name=="PPO":
        model_PPO = PPO("MlpPolicy", env, verbose=1,seed=1, tensorboard_log = dossier+"/"+env_name+"_PPO_"+str(number_it)+"/")
        model_PPO.learn(int(number_it))
        model_PPO.save(dossier+"/"+model_name+"_"+env_name+"_"+str(number_it))
  
# Graphique 1 (épisode court)
training(env, "highway_fast_env", "A2C", "graphique_l1", 200)
# training(env, "highway_fast_env", "DQN", "graphique_l1", 50000)
# training(env, "highway_fast_env", "PPO", "graphique_l1", 50000)

# # Graphique 2 (épisode long)
# training(env_long, "highway_fast_env_long", "A2C", "graphique_l2", 50000)
# training(env_long, "highway_fast_env_long", "DQN", "graphique_l2", 50000)
# training(env_long, "highway_fast_env_long", "PPO", "graphique_l2", 50000)

# # Graphique 3 (différente densité)
# training(env_long   , "highway_fast_env_long"   , "DQN", "graphique_l3", 50000)
# training(env_long_d2, "highway_fast_env_long_d2", "DQN", "graphique_l3", 50000)
# training(env_long_d3, "highway_fast_env_long_d3", "DQN", "graphique_l3", 50000)   
# training(env_long_d4, "highway_fast_env_long_d4", "DQN", "graphique_l3", 50000)  
# training(env_long_d5, "highway_fast_env_long_d5", "DQN", "graphique_l3", 50000)  

# # Graphique 4 (Essai pour corriger la solution: j'essai densité = 3)
# training(env_long_d3_speed, "highway_fast_env_long_d3_speed", "DQN", "graphique_l4", 50000)
# training(env_long_d3_collision, "highway_fast_env_long_d3_collision", "DQN", "graphique_l4", 50000)
# training(env_long_d3_lane, "highway_fast_env_long_d3_lane", "DQN", "graphique_l4", 50000)   

# training(env_long, "highway_fast_env_long", "DQN", "graphique_l5", 20000)

        
        
        
        