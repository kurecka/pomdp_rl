This is a simple reimplementation of STORM POMDPs in pytorch. The goal is to have a simple and easy to understand codebase which would allow one to quickly implement POMDPs and train rl agents on them in a vectorized manner.

## Implemented POMDPs
- Evade
- Refuel

## Implemented Algorithms
- PPO (memoryless)
- PPO with shared model (memoryless)
- LSTM PPO
- DQN (memoryless)
- DDQN (memoryless)

## How to run
```bash
pip install -r requirements.txt
pip install -e .

python evade_playground.py  # to run the Evade POMDP with PPO. Uncomment the desired configuration in the file.
python -m pytest --disable-warnings  # to run tests
```
