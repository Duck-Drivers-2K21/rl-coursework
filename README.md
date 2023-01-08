# Playing Pong with Deep Reinforcement Learning

An exploration in Deep Reinforcement Learning by writing a range of agents that learn to play pong.

### Agents
- Deep Q-Networks (agents/DQN)
- Double DQN (agents/DDQN)
- Proximal Policy Optimisation (agents/PPO.py)

### Baselines
- Random Move Baseline (BaselineAgents.py)
- Ball Tracker Baseline (BaselineAgents.py)

## How to run

### Set up
First you will need to install the requirements.

For NVIDIA systems, run:

```pip install -r requirements.txt```

For non-NVIDIA systems, run:

```pip install -r requirements-no-nvidia.txt```

Then run:
```AUTOROM```

### Running Baseline Agents:
Our baseline agents use a `ram` representation of the environment. See `Pong.py` for details on how they interact with the environment. The logic for both agents can be found in `BaselineAgents.py`.

You can run the baseline agents with this command: `python main.py`.


### Running RL Agents:
The RL agents we have implemented (DQN, DDQN and PPO) use an `rbg_array` representation of the environment. Each agent's logic is placed on a separate file inside of the `agents/` directory.

To run an agent, simply run the corresponding `.py` file. E.g. to run our DQN agent: `python agents/DQN.py`. See Agents list for details.
