# pong

An exploration in Reinforcement Learning, by writing a range of Agents that learn to play pong.

## Setup

RUN:

```pip install -r requirements.txt```

THEN:

```AUTOROM```


### Running Baseline Agents:
Our baseline agents use a `ram` representation of the environment. See `Pong.py` for details on how they interact with the environment. The logic for both agents can be found in `BaselineAgents.py`.

You can run the baseline agents with this command: `python main.py`.


### Running RL Agents:
The RL agents we have implemented (DQN, DDQN, Rainbow and PPO) use an `rbg_array` of the environment. Each agent's logic is placed on a separate file inside of the `agents/` directory.

To run an agent, simply run the corresponding `.py` file. E.g. to run our DQN agent: `python agents/DQN.py`.
