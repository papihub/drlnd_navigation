# drlnd_navigation
Deep Reinforcement Learning - Project Navigation

![Agent in banana environment](https://video.udacity-data.com/topher/2018/June/5b1ab4b0_banana/banana.gif)

This repository contains a Deep Reinforcement learning agent. This agent navigates an square environment with yellow and blue bananas. The goal of the agent is to collect as many yellow bananas as possible and as few blue bananas as possible. The agent receives a reward of +1 for every yellow banana it collects and -1 for every blue banana it collects.

## Dependencies
I developed code in this repository on a windows 10 64bit OS. So, I havent tested if this code works on any other OS.

**Miniconda**: Install miniconda3 from [miniconda download page](https://docs.conda.io/en/latest/miniconda.html)

**Python**: Follow the instructions in [DRLND Github repository](https://github.com/udacity/deep-reinforcement-learning#dependencies) to set up your python environment. These instructions will guide you to install PyTorch, ML-Agents toolkit and a couple of other python packages required for this project.

**Unity Environment**: Download the unity environment from [here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip). This is for windows 10 64bit OS. Please refer to the course material if you want the environment for a different OS.

## Environment
The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around the agent's forward direction. Given this information, the agent has to learn how to best select actions. Four discrete actions are available, corresponding to:
**0** - Move Forward
**1** - Move Backward
**2** - Turn Left
**3** - Turn Right

The task is episodic. This means there is a distinct "done" state. In order to solve the environment, our agent must get an average of +13 reward over 100 consecutive episodes.

## Success criteria:
Our agent must get an average of +13 reward over 100 consecutive episodes

## Instructions to train the agent:
Install the dependencies above.

open a jupyter notebook.

Run the notebook navigation.ipynb to train and test the agent. The notebook has instructions to load a saved model and to save a trained model.

## Approach : Reinforced Learning with Deep Q networks
This is agent trains using reinforcement learning algorithms and deep neural networks. 

Initially the neural networks are initialized with random weights.

The agent interacts with the environment by taking actions and observing the reward and state changes.

When the agent takes an action in a current state, the environment returns the next state along with a reward and a *done* indicator if we reached the terminal state.

When the agent reaches the terminal state, the eposide terminates.

The agent maintains a tuple of ( state, action, reward, next state, done ) for all steps in an episode in a replay buffer.

The agent maintains as many episodes as its replay buffer can fit. In this case the replay buffer is set to 1e5 bytes.

The agent samples episodes from the replay buffer and trains its neural networks models.

The agent does not learn after every step. Instead it learns once every 4 steps ( hyper parameter )

The agent uses its deep neural netowrk to come up with a greedy action given the current state. It uses an epsilon greedy method to either pick a random state ( to explore ) or a greedy action ( to exploit ).

epsilon is initially set to a high value of 1 to encourage exploration during the initial steps. Then its gradually reduced till it reaches 0.01 and maintains at 0.01. This is to allow the agent to be greedy over time as it has learned from the environment.

For stability of the neural networks, the agent maintains a target network and a local network. The local network is used to take actions and is refeined in each learning step. The target network is only updated after a fixed number of learning steps.

In each learning step, the agent computes the difference between expected and predicted value and uses a learning rate along with a discount factor. It uses this difference (loss) to train the deep network.

### Network Architecture
The agent uses 2 identical deep neural networks to learn from the environment interactions.

The networks have 1 input layer, 1 hidden layer and 1 output layer.

All layers are fully connected.

Input layers has (state_size) inputs and (state_size)x10 outputs

Hidden layer has (state_size)x10 inputs and (state_size)x5 outputs

output layer has (state_size)x5 inputs and (action_size) outputs

State_size = 37 and action_size = 4 for this environment.

Input and hidden layers go thru an relu activation function.

We use a mean squared loss function to compute the loss values.

We use Adam optimizer to backpropogate the loss and update weights.

### Hyper parameters and their values
|Hyper parameter|Value|Comment|
|---------------|:---:|-------|
|Replay buffer size|1e5|BUFFER_SIZE|
|Discount Factor|0.99|GAMMA|
|How often do we learn?|4|UPDATE_EVERY|
|Minimum steps before we start learning|64|BATCH_SIZE|
|Factor for target network update|1e-3|TAU|
|Learning Rate|5e-4|ALPHA/LR|
|Initial epsilon|1.0|for exploration|
|Min epsilon|0.01||
|epsilon decay|0.995||
|*below are other parameters used in training the agent*|
|Max episodes|2000||
|Max steps in an episode|1000||
|window size to compute average score|100||

## Future actions / steps
In this implementation I sampled randomly from the replay buffer for training.
Implementing priority queues can allow some of the agent to learn from important steps more times than other steps.
Implement dual networks where we switch the target and local networks.
Learn directly from raw images than from encoded states!!
