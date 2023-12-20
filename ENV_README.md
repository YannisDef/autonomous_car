# Comment changer l'environnement

## Résumé

Dans le projet que nous avons confectionné, l'environnement de "highway_env" n'est pas parfait. Pour refaire les mêmes expériences que nous, vous devez modifier un fichier pour que l'agent puisse de pas forcément avancer.

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
Une fois cela fait, vous devez appeler cette fonction, implémenté juste au-dessus pour overwrite le fichier de la lib par le vôtre pour pouvoir modifier les vitesses de l'agent.
```
overwrite_lib(source_file, destination_file)
```