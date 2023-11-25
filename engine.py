import gymnasium as gym

env = gym.make('highway-v0', render_mode='human')
# env = gym.make('merge-v0', render_mode='human')
# env = gym.make("racetrack-v0", render_mode='human')
# env = gym.make('intersection-v0', render_mode='human')
# env = gym.make('parking-v0', render_mode='human')
# env = gym.make('roundabout-v0', render_mode='human')

obs, info = env.reset()
done = truncated = False
while not (done or truncated):
    action = ... # Your agent code here
    obs, reward, done, truncated, info = env.step(action)