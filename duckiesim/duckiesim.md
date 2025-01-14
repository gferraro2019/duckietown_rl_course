# Duckiebot in Simulation

## Python setup:

### 0. Create the conda environment:
Let's start by setting up the Python environment required for this project. Use the following command to create the conda environment:

`conda env create -f environment.yaml`

### 1. Activate the new conda environment:
After creating the environment, activate it using:

`conda activate duckietownrl`

### (read twice) To remove the conda environment:
If you want (at some point) to remove the environment later, use:

`conda remove -n duckietownrl --all`

## First steps with the simulator:

### For playing with the keyboard:
To control the Duckiebot manually using your keyboard, run:

`python manual/manual_control.py`

## Codebase overview:


## Reinforcement Learning:
Now it's time to train the Duckiebot to drive autonomously! We will use reinforcement learning to train the Duckiebot to follow the lane. The code for training the Duckiebot is in the `rl` directory.

### Environment presentation:

### Environment interaction:

### Training the Duckiebot:
