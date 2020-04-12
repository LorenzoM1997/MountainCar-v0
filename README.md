# MountainCar-v0
Reinforcement Learning algorithm to solve MountainCar-v0 in OpenAI gym. It uses
the DQN agent by tf-agents library

## Set up
The environment is created and run in a docker container. To create the docker
image with all requirements, you can run
```
make docker
```

## Running the environment
To run the environment you can exec into the container and the run the files. To
mount the working environment and open the bash inside the container you can run
```
make exec
```

### Train the model
The training loop is in the file `MountainCar-v0.py`. To run it you can just use
the command
```
python3 MountainCar-v0.py
```

### Visualize
To create a graph of the average evaluation reward over time you can run
```
python3 visualization.py
```
