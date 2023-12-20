# Comment changer l'environnement et tester notre projet

## Résumé

Dans le projet que nous avons confectionné, l'environnement de "highway_env" n'est pas parfait. Pour refaire les mêmes expériences que nous, vous devez modifier un fichier pour que l'agent puisse avoir une vitesse minimum intéressante.

Actuellement, la vitesse de l'agent est comprise entre 20 et 30 unités. Si l'agent est devant un bouchon est qu'il ne peut emprunter aucune voie, il faut que celui-ci s'arrête. Sans cette modification, ca sera impossible pour lui.

### 0 - Cloner le repo
Tout d'abord il faudra cloner le repo.
```
git clone git@github.com:YannisDef/autonomous_car.git
```

### 1 - Installer highway_env
Ensuite, il faut installer l'environnement
```
pip install highway-env
```

### 2 - Configuration de l'environnement
Voici ce qu'il faut faire dans le fichier où vous voulez refaire l'expérience.

#### 2.a Outils de setup
Ces lignes sont à simplement copier coller. Vous pouvez modifier les variables de speed pour lui donner une limite de vitesse minimale et maximale.
```
import gymnasium as gym
from stable_baselines3 import DQN
import numpy as np

speed_list = []

speed = {
    'min': 0,
    'max': 30
}

from controller import *

def overwrite_lib(source, destination):
    try:
        with open(source, 'r') as fichier_source:
            contenu = fichier_source.read()

        with open(destination, 'w') as fichier_destination:
            fichier_destination.write(contenu)

        print(f"Succès: Le contenu de '{source}' a été copié avec succès vers '{destination}'.")
    except FileNotFoundError:
        print("Erreur: Le fichier source n'a pas été trouvé.")
    except Exception as e:
        print(f"Erreur: Une erreur s'est produite: {e}")
```

#### 2.b Récuperer les paths des fichiers à échanger
Le premier est le path jusqu'au fichier "controller.py" dans le repo actuel que vous venez de clone.
```
source_file = 'YOUR/PATH/autonomous_car/controller.py'
```

Le second path correspond à celui du fichier "controller.py" dans "highway_env" que vous avez installé dans les étapes précédentes.
```
destination_file = 'YOUR/PATH/highway_env/vehicle/controller.py'
```

#### 2.c Overwrite le fichier de base par le votre
Une fois cela fait vous devez appeler cette fonction, implémenté juste au-dessus, pour overwrite le fichier de la lib par le vôtre pour pouvoir modifier les vitesses de l'agent.
```
overwrite_lib(source_file, destination_file)
```

### 3 - Tester
Voici un exemple de code pour tester le modele donné en exemple.
```
import gymnasium as gym
from stable_baselines3 import DQN

#
INSERER ICI LA PARTIE POUR OVERWRITE LA LIB
#

# Configurer l'environnement
env = gym.make("highway-v0", render_mode='human')
env.config["vehicles_density"] = 1
env.config["duration"] = 300
env.config["collision_reward"] = -10
env.config["vehicles_count"] = 40

# Load le modele
model = DQN.load("train_frequency/model_density_1_duration_300_collision_m10_vehicles_count_40_sreward_0_tr_fr_2_episode.zip")

# Observer l'agent
while True:
  done = truncated = False
  obs, info = env.reset()
  while not (done or truncated):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, info = env.step(action)
    env.render()
```

### 4 - Développer
Si vous voulez développer votre propre IA, rien ne vous empeche d'utiliser la fonction "run_and_save_agent" pour entrainer et enregistrer votre agent.
```
def run_and_save_agent(
    density,
    b_size,
    duration,
    col_r,
    v_count,
    speed_reward,
    tr_fr,
    nb_steps,
    path_to_save,
):
    env.config["vehicles_density"] = density
    env.config["duration"] = duration
    env.config["collision_reward"] = col_r
    env.config["vehicles_count"] = v_count
    env.config["high_speed_reward"] = speed_reward

    model = DQN(
        "MlpPolicy", env,
        policy_kwargs=dict(net_arch=[256, 256]),
        learning_rate=5e-4,
        buffer_size=15000,
        learning_starts=200,
        batch_size=b_size,
        gamma=0.8,
        train_freq=tr_fr,
        gradient_steps=1,
        target_update_interval=50,
        verbose=1,
        tensorboard_log=path_to_save,
        seed=1
    )
    model.learn(nb_steps)
    model.save(path_to_save)

# Change parameters to test some new environnements
run_and_save_agent(1, 32, 100, -1000, 35, 2, (1, "step"), 30000, PATH)
```
Vous pourrez ensuite (ou en temps réél) observer les résultats de votre agent avec la commande.

```
$> tensorboard --logdir=path_to_save/
```


