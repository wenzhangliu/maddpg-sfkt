# Efficient Exploration for Multi-agent Reinforcement Learning via Transferable Successor Features
Multi-agent reinforcement learning algorithms: MADDPG-SFs & MADDPG-SFKT

This is an open-source code for our research work about multi-agent reinforcement leanring. 

# Requirements: 
- Python 3.6

# Dependencies
- Tensorflow 1.5 + Gym

# Usage

## Experiment 1: Cooperaitve box-pushing environment (source domain)

- Train:
  
  "python main_sfs.py --iftrain 1 --scenario "simple_push_box_multi" --method MADDPG-SFs --penalty -0.1"
  
- Test:
  
  "python main_sfs.py --iftrain 0 --scenario "simple_push_box_multi""

## Experiment 1: Cooperaitve box-pushing environment (target domain)

- Train: Knowledge transfer and fine tune. 
  
  "python main_transfer.py --iftrain 0 --istransfer 1 --scenario "simple_push_box_multi" --penalty -1.0"

  "python main_transfer.py --iftrain 1 --istransfer 0 --scenario "simple_push_box_multi" --penalty -1.0"

- Test:
 
  "python main_transfer.py --iftrain 0 --istransfer 0 --scenario "simple_push_box_multi""

## Experiment 2: Non-monotonic predator-prey environment (source domain)

- Train:
  
  "python main_sfs.py --iftrain 1 --scenario "predator_prey" --method MADDPG-SFs --penalty1 -0.0 --penalty2 -0.0 --prey-policy random"

- Test:
  
  "python main_sfs.py --iftrain 1 --scenario "predator_prey" --method MADDPG-SFs --penalty1 -0.0 --penalty2 -0.0 --prey-policy random"

## Experiment 2: Non-monotonic predator-prey environment (target domain)

- Train: Knowledge transfer and fine tune.
  
  "python main_transfer.py --iftrain 0 --iftransfer 1 --scenario "predator_prey" --method MADDPG-SFKT --penalty1 -1.0 --penalty2 -1.0 --prey-policy random"
  
  "python main_transfer.py --iftrain 1 iftransfer 0 -scenario "predator_prey" --method MADDPG-SFKT --penalty1 -1.0 --penalty2 -1.0 --prey-policy random"

- Test:
 
  "python main_transfer.py --iftrain 0 iftransfer 0 -scenario "predator_prey" --method MADDPG-SFKT --prey-policy random"
