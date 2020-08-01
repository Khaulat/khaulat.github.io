---
Title: "REINFORCE"
Date: 2020-08-01
Tags: [machine learning, reinforcement learning, deep learning]
header:
  image: "/images/reinforce.png"
excerpt: "My implementation of the "REINFORCE" reinforcement learning algorithm"
---


# Understanding REINFORCE

An agent was trained in the OpenAI gym's pong environment to move left or right and hit a ball. This algorithm is a simple implementation of a policy-based method for training agents in reinforcement learning. Policy-based methods try to find the policy used for choosing actions by the agent directly instead of first evaluating for an intermediate step -> value-functions as done in [Deep Q-Networks](https://khaulat.github.io/Deep-Q-Networks(DQN)/).

The REINFORCE algorithm uses a function approximator(neural network) to predict a policy given the state. To get the optimal policy, the agent's **goal** is to estimate for the best weights used at each neuron with hill-climbing - also known as **gradient ascent** that would maximize the expected return.

It works with the aim of making all actions that lead to the agent winning more likely to occur while those that lead to the agent losing less likely to occur.


## Learning Algorithm

As mentioned above, the agent uses gradient ascent to optimize the weights in other to get the best policy. In this case, the agent steps in the direction of the gradient instead of the opposite as in gradient descent. To get the estimate of the gradient and update the policy, the REINFORCE algorithm typically follows these 3 steps;

- Uses the initial policy to collect sample trajectories.
- Sample trajectories are then used to estimate the gradient.
- The estimated gradient is then used to update the weights of the policy.

The agent was trained for a maximum of 1000 episodes.

<img src="{{ site.url }}{{ site.baseurl }}/images/reinforce_alg.png" alt="The REINFORCE Algorithm">


## Methods

The model was trained with 2 convolutional layers and 2 fully connected layers with a RELU activation function. Two layers were stacked in order to get a better prediction. A sigmoid function was then used in the final layer to predict the probability of selecting an action - in this case, left.
The **`Control.ipynb`** contains the code for training and evaluation, **`model.py`** contains the Q-Network architecture while **`ddpg_agent`** contains the reinforcement learning algorithm.


## Results

The agent got better at playing when compared to its initial play before training! There is a possibility of it improving with additional training episodes.


## Future Improvements

Even though this algorithm performed quite well, it still suffers from some obvious challenges;

- The update process is very inefficient as we run the policy once, update once, and then throw away the trajectory.
- The gradient estimate is very noisy. There is the possibility of collecting trajectories that may not be representative of the policy.
- There is no clear credit assignment. The fact that an action leads to a win doesn't mean it is the best action and the fact that it leads to a loss doesn't mean it is a bad action. The dependence of the credit on the final actions is not clear.

Fortunately, there are solutions to these problems! One method that directly solves for the problems mentioned above is the **Proximal Policy Optimization method**. Read more about it [here]().

The **Actor-Critic method** is also another great improvement over the REINFORCE algorithm. This method combines both policy-based and value-based methods - solves the bias problem in value-based methods and the variance problem in policy-based methods.

The code to this article can be found [here](https://github.com/Khaulat/Deep_Reinforcement_Learning/tree/master/PONG_with_REINFORCE). Check for better understanding.


