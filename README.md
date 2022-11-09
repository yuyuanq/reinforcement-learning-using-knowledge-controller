# Reinforcement Learning Using Knowledge Controller

Implementation for "Accelerating Deep Reinforcement Learning via Knowledge-Guided Policy Network".


## Setup Environment

### Environement Requirements
* python 3.7
* pytorch 1.7
* gym 0.16.0
* wandb
* tensorboardX

## Our Network Structure and Rules

See ```Controller``` in ```controller.py``` and ```XXXRule``` in ```./rule/*.py``` for details.

### Training

All log and snapshot would be stored logging directory. Logging directory is default to be "./output/ENV_NAME". Different environments can be used by providing env values in the args.

```
# For CartPole without Knowledge
python main.py --env CartPole-v1 --no_controller --max_update 5000 --seed 0

# For CartPole with Knowledge
python main.py --env CartPole-v1 --max_update 5000 --seed 0

# For FlappyBird without Knowledge
python main.py --env FlappyBird --no_controller --max_update 1200 --seed 0

# For FlappyBird with Knowledge
python main.py --env FlappyBird --no_controller --max_update 1200 --seed 0

```