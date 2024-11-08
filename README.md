## DUCKIETOWNRL:
A Reinforcement Learning Course with DuckieTown.

## INSTALLATION:

### 0. Create the conda environment:
`conda env create -f environment.yaml`

### 1. Activate the new conda environment:
`conda activate duckietownrl`

## UNINSTALLATION:

### To remove the conda environment:
`conda remove -n duckietownrl --all`

## SIMULATOR:
### For playing with the keyboard:
`./manual_control.py`

### For playing with the joystick:
`./joystick_control.py`

## TRAINING A MODEL WITH SAC:
The following script can run several environments in parallel and collect experience in the same Replay Buffer

### FOR A STANDARD VERSION:
`python duckietownrl/parallel_training.py`

### FOR IMITATION LEARNING WITH JOYSTICK:
To activate and deactivate the joystick control, press the j key. By default, the joystick is activated, and you have to move with it to collect a new experience; otherwise, it returns, and no steps are done. Once you finish manually controlling, press the j key to let the agent train on his own.

`python duckietownrl/parallel_training_imitation_learning.py`

## TO EVALUATE A MODEL:
Remember to specify the path for your model in the script.

`python duckietownrl/evaluate.py`
