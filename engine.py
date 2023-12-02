import gymnasium as gym
import highway_env
from stable_baselines3 import DQN
import json
import pickle
import numpy

# # ML sur un env avec beaucoup de vehicule et densité faible.

def setup_environment():
  with open('setup.json') as f:
    data = json.load(f)
    env.config["vehicles_count"] = data["vehicles_count"]
    env.config["vehicles_density"] = data["vehicles_density"]
    env.config["controlled_vehicles"] = data["controlled_vehicles"]
    env.config["manual_control"] = data["manual_control"]

class Agent:
  def __init__(
    self,
    env,
    name: str = 'Patrick',
    seed:float = 1.,
    learning_rate: float = 1.,
    crawl_rate: float = 1.
  ):
    """ Agent who use (?) algorithm

    Args:
        learning_rate (int, optional): taux d'apprentissage. Defaults to 1.
        crawl_rate (int, optional): taux d'exploration. Defaults to 1.
    """
    # use the seed depend of what lib we will use
    self.env = env
    self.learning_rate = learning_rate
    self.crawl_rate = crawl_rate
    self.q_s_a = numpy.ones((3, 3))
    self.pi = numpy.ones((3, 3)) / 3

    self.name = name
    self.historical = []

  def __str__(self):
    r = "Agent: " + self.name + \
      "\nlearning_rate: " + str(self.learning_rate) + \
      "\ncrawl_rate: " + str(self.crawl_rate) + \
      "\n"
    return r

  def predict(self, observations):
    pass

  def run_one_session(self, observations) -> tuple[int, list]:
    """ Run only one session

    Returns:
        reward: What the agent won after the session
        historical: All choices and rewards of the session
    """
    historical = []
    done = truncated = False
    while not (done or truncated):
      action, _states = model.predict(observations, deterministic=True)
      observations, reward, done, truncated, info = env.step(action)
      historical.append(action, reward)
      env.render()
    return reward, historical

  def reward_agent(self, reward):
    """_summary_

    Args:
        reward (_type_): _description_
    """
    pass

  def run(self, cycles=1000):
    """ Run a session during 'cycles' times 

    Args:
        cycles (int, optional): Number of time you want run a session. Defaults to 1000.
    """
    for cycle in range(cycles):
      observations, informations = self.env.reset()
      reward, historical = self.run_one_session(observations)
      self.historical.append(historical)
      self.reward_agent(reward)

  def save_model(self, agent_name: str):
    """ Save the model to use him in an other runing time

    Args:
        agent_name (str): name of the agent
    """
    with open(agent_name + '.plk', 'wb') as file:
        pickle.dump(var, file)

  def load_model(self, agent_name: str):
    """ Load the model from agent name

    Args:
        agent_name (str): Name of the agent to load (previously saved whit save_model() function)

    Returns:
        Agent: The agent
    """
    with open(agent_name, 'rb') as file:
      agent = pickle.load(file)
    return agent

env = gym.make("highway-fast-v0", render_mode='human')
# env = gym.make("merge-v0", render_mode='human')
setup_environment()

model = Agent(
  env,
  name='Federer',
  seed=1,
  learning_rate=1,
  crawl_rate=1
)

model.run()
