---
Title: "Proximal Policy Optimization"
Date: 2020-08-01
Tags: [machine learning, data science, deep-learning, neural-networks, reinforcement-learning, RL, deep-reinforcement-learning]
header:
  image: "/images/ppo.png"
excerpt: "My implementation of the Proximal Policy Optimization Algorithm"
---

# Understanding Proximal Policy Optimization

This algorithm is a direct improvement over the REINFORCE algorithm. The problems faced by REINFORCE as stated in this [article](https://khaulat.github.io/Reinforce-Algorithm/) include;

- The update process is very inefficient as the policy is run only once, updated once, and then the trajectory is thrown away.
- The gradient estimate is very noisy. There is the possibility of collecting trajectories that may not be representative of the policy.
- There is no clear credit assignment. The fact that an action leads to a win doesn’t mean it is the best action and the fact that it leads to a loss doesn’t mean it is a bad action. The dependence of the credit on the final actions is not clear.

The same PONG environment as used to implement the REINFORCE algorithm is used here.


## Learning Algorithm

As with the REINFORCE algorithm, the agent uses gradient ascent to optimize the weights in order to get the best policy. In this case, the agent steps in the direction of the gradient instead of the opposite as in gradient descent. To get the estimate of the gradient and update the policy, the REINFORCE algorithm typically follows these 3 steps;

- Uses the initial policy to collect sample trajectories.
- Sample trajectories are then used to estimate the gradient.
- The estimated gradient is then used to update the weights of the policy.

The agent was trained for a maximum of 1000 episodes.


## Methods

We solve for the **`first problem`** by reusing trajectories instead of disposing of them after learning. This is done via **`importance sampling`**. Importance sampling recycles used trajectories for future training and makes policy update much more efficient. For any given policy a set of trajectories is selected with a probability, P. The probability of selecting this same set of trajectories for another policy would be different. In order to keep selecting the same trajectories across all policies, we multiply them by a **reweighting factor**. This is what *importance sampling* does - helps select the best reweighting factors that would keep the trajectories stable.


To solve for the **`second problem`** which is reducing the noise while estimating for the gradient, we remove factors that are not very significant to setting the trajectories. For example, if the old and current policy is close enough to each other, there would be some common factors  that are pretty close to 1, which would be ignored.

After removing these unnecessary factors, the new gradient is known as the surrogate function. Gradient ascent is then performed on this surrogate function to update the policy.


Another problem we can face involves *`poor approximations of the policies`*. In order to avoid this, we clip the surrogate function to ensure that the new policy remains close to the old one. This problem occurs during gradient ascent at some point when we hit a cliff, the policy changes by a large amount. From the perspective of the surrogate function, the average reward is really great, but the actual average reward is really bad!

<img src="{{ site.url }}{{ site.baseurl }}/images/clipped-surrogate.png" alt="Description of the clipped surrogate function">


For the **`third problem`**, not just the final result matters but the results at various intervals during learning.


## Results

Some improvement was definitely made over the REINFORCE algorithm.


## Future Improvements

Using a value-function as a baseline to reduce the variance of such policy-based method as used in Actor-Critic methods is one improvement to consider that might help produce better actions.


**Code** to this article can be found [here](https://github.com/Khaulat/Deep_Reinforcement_Learning/tree/master/PONG_with_PPO). Check for better understanding.


