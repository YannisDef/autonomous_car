# model = DQN('MlpPolicy', env,
#               policy_kwargs=dict(net_arch=[256, 256]),
#               learning_rate=5e-4,
#               buffer_size=15000,
#               learning_starts=200,
#               batch_size=32,
#               gamma=0.8,
#               train_freq=1,
#               gradient_steps=1,
#               target_update_interval=50,
#               verbose=1,
#               tensorboard_log="highway_dqn/")
# model.learn(int(2e4))
# model.save("highway_dqn/model")

# model = DQN.load("highway_dqn/model")

# while True:
#   done = truncated = False
#   obs, info = env.reset()
#   while not (done or truncated):
#     action, _states = model.predict(obs, deterministic=True)
#     obs, reward, done, truncated, info = env.step(action)
#     env.render()

# ----------------------------------------------------------------

# import gymnasium as gym

# # env = gym.make('highway-v0', render_mode='human')
# # env = gym.make('merge-v0', render_mode='human')
# # env = gym.make("racetrack-v0", render_mode='human')
# # env = gym.make('intersection-v0', render_mode='human')
# # env = gym.make('parking-v0', render_mode='human')
# # env = gym.make('roundabout-v0', render_mode='human')

# obs, info = env.reset()
# done = truncated = False
# while not (done or truncated):
#     action = ... # Your agent code here
#     obs, reward, done, truncated, info = env.step(action)


# https://highway-env.farama.org/quickstart/?fbclid=IwAR1PAVIw7bEUiaTlyzF4cOQn9QpNcx3oze6ouZ9Wif5Q0Wp842TLsvqYrvI
# https://github.com/Farama-Foundation/HighwayEnv
