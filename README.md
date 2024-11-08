## DUCKIETOWNRL:
A Reinforcement Learning Course with DuckieTown.

## INSTALLATION:

### 0. Create the conda environnment:
`conda env create -f environment.yaml`

### 1. Activate the new conda environnment:
`conda activate duckietownrl`

## UNINSTALLATION:

### To remove the conda environnment:
`conda remove -n duckietownrl --all`

## SIMULATOR:
### For playing with the keyboard:
`./manual_control.py`

### For playing with the joystick:
`./joystick_control.py`

## TRAINING A MODEL WITH SAC:
The following script can run several environments in parallel and collecte experience in the same Replay Buffer

### FOR A STANDARD VERSION:
`python duckietownrl/parallel_training.py`

### FOR IMITATION LEARNING WITH JOYSTICK:
To activate and desactivate the control with the joystic press the j key. By default the joystick is activated and you have to nove with it to collect a new experience otherwhise it returns and no step are done. Once you finished to manually control press the j key to let the agent training on his own.

`python duckietownrl/parallel_training_imitation_learning.py`

## TO EVALUATE A MODEL:
Remember to specify the path for your model in the script.

`python duckietownrl/evaluate.py`
