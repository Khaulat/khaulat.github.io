---
Title: "Understanding Deep Q-Learning"
Date: 2020-07-14
Tags: [machine learning, data science, q-learning, deep-learning, neural-networks, reinforcement-learning, rl, deep-reinforcement-learning]
header:
  image: "/images/deep_ql.png"
excerpt: "This is a report for a Reinforcement Learning project - An agent trained to navigate within a Unity environment"
---

# Understanding Deep Q-Learning

In this article, I aim at explaining the algorithm behind [this project](https://github.com/Khaulat/Deep_Reinforcement_Learning/tree/master/Navigation_project). As navigation is one very important activity performed by every moving object, the project teaches a reinforcement learning agent to navigate through a field layed with bananas; some yellow and some blue. The goal of the agent is to pick as many yellow bananas as possible while ignoring blue bananas.

Like any other typical reinforcement learning problem, the 'environment', 'state', 'rewards', 'observations' and 'actions' need to be defined. 

- **Environment** - This is the area where the agent navigated through and learns. For this project, a Unity environment is used.
- **State** - The current status of the agent. The state space for this project has 37 dimensions. The agent could be in any of the states at any time.
- **Observation** - This is what the agent sees in a environment. Here, the agent observes a local ray-cast perception of nearby objects.
- **Action** - Whatever step the agent takes in a given state. For this project, the agent can decide to move forward, backward, turn left or right.
- **Reward** - The price gotten for taking any action; whether right or wrong. The agent gets a **+1** reward for colliding with a yellow banana and a **-1** reward for colliding with a blue banana.


<img src="{{ site.url }}{{ site.baseurl }}/images/rl.jpg" alt="A visual description of an RL problem">


To train RL agents with the DQN algorithm, we need to define an action-value fuction and try to improve the policy based on this function. This action value function is the estimated reward by the agent for a given state based on an action taken. We update/improve the policy in order to get the best set of actions to be performed by an agent in order to reach its goal.

## Learning Algorithm

A modified version of Deep Q-Networks was used for learning - Double DQN(DDQN) algorithms. Unlike the DQN algorithm that focus on selecting the actions that maximizes the values, which leads to an estimation that is not robust, DDQN solves this problem of over estimation of action-values. 
It proposes a solution that works well in practice by using different parameters for selecting actions and evaluation.

The task is designed to be episodic - terminates after a given number of episodes. The agent was trained for a maximum of 2000 episodes or until the environment was solved. Each of the episodes made up of 2000 time-steps. It stores its experience from each timestep in a replay buffer. This experience is a combination of the current state and chosen action, as well as the resulting reward and next state of the environment. The agent interacts with the environment according to an e-greedy policy which allows it to select other actions asides those that maximize the action-value. This allows the agent to learn more about the environment by exploring. Parameters used to store and sample from the replay buffer include;

- **'BUFFER_SIZE'** '10^5' - This is the size of the replay buffer. Specifies how much experiences should be stored.
- **'BATCH_SIZE'** '32' - The amount of experiences we take from the replay buffer at a time to train the agent.
- **'UPDATE_EVERY'** '4' - The specifies the amount of timesteps it takes to sample a batch from the replay buffer.
- **'LR'** '5.10^-4'- The speed at which the agent learns from experiences.

As mentioned above, DDQN uses a different Q-Network for selecting actions and a different one for evaluating. Two Q-Networks(Local and Target Networks) with the same achitecture but different weights serve this purpose.
The local network is trained by minimizing the mean-squared loss function and used the Adam optimizer with a learning rate of **LR**, then the target network is  updated(*soft update*) using the value of **TAU** towards the local network.


## Methods

The model has 37 input neurons which are the state-space dimensions, three fully-connected layers with 64,32, and 16 neurons respectively with ReLU activation functions, and a linear output layer with four neurons that represent the state-action value for each possible action.
The **'run_navigation.ipynb'** contains the code for training and evaluation, **'NN_Model.py'** contains the Q-Network architecture while **'rl_algorithm'** contains the reinforcement learning algorithm.

After tuning, the hyperparameters that prove to fit the model well are;

| Hyperparameter | Value |
| ----------- | ----------- |
| BUFFER_SIZE | 10^5 |
| BATCH_SIZE  | 32 |
| UPDATE_EVERY | 4 |
| GAMMA | 0.95 | 
| LR | 5.10^-4 |
| TAU | 5.10^-2|
| start_epsilon | 1.0 |
| epsilon_min | 0.1 |
| epsilon_decay | 1000 |


For the e-greedy policies used to select actions, the value is initialized at the beginning of training in **'start_epsilon'**, it is then annealed to a value of **'epsilon_min'** within **'epsilon_decay'** episodes. Linear annealing performs better when compared to exponential decay making it the choice for the algorithm.

## Results

The DDQN agent was able to solve the environment after 944 episodes with an average score of +13, however, there's still room for improvements! The corresponding model weights are stored in ./checkpoints/checkpoint888.pth.  

## References

- [Human-level control through deep reinforcementlearning](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf)
- [Deep Reinforcement Learning with Double Q-learning](https://arxiv.org/abs/1509.06461)
- [Issues in Using Function Approximation for Reinforcement Learning](https://www.ri.cmu.edu/pub_files/pub1/thrun_sebastian_1993_1/thrun_sebastian_1993_1.pdf)
