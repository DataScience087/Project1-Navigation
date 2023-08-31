# Learning Algorithm

This project was solved using a Deep Q-Network learning algorithm. Deep Q-Networks involve alternating between sampling the environment and learning.

The sampling the environment step involves taking epsilon greedy actions A with respect to the current estimate of q from state S. The environment returns back a reward R and next state S'. The (S,A,R,S') tuples are stored in replay memory D. 

The learning step involves taking random minibatch of tuples from D and learning from the batch using gradient descent to update the weights in the neural network. Note this implementation used the Fixed Q-Targets technique so two sets of weights need to be maintained. 

## The hyperparameters chosen are as follows:
- n_episodes=2000 (number of episodes for which to train)
- max_t=1000 (maximum number of time steps per episode)
- eps_start=1.0 (max value of epsilon)
- eps_end=0.01 (min value of epsilon)
- eps_decay=0.995 (value of (epsilon at episode n+1)/(epsilon at episode n))
- BUFFER_SIZE = int(1e6)  (size of the replay buffer)
- BATCH_SIZE = 64         (number of tuples to sample from memory during a learning step)
- GAMMA = 0.99            (discount factor of reward at t+1 to reward at t)
TAU = 1e-3              # for soft update of target parameters
LR = 5e-4               # learning rate 
UPDATE_EVERY = 4        # how often to update the network

The report clearly describes the learning algorithm, along with the chosen hyperparameters. It also describes the model architectures for any neural networks.

## Plot of Rewards

As shown in this plot, the trained agent is able to receive an average reward (over 100 episodes) of at least +13. 

![AverageScoreLast100Episodes](https://github.com/DataScience087/Project1-Navigation/assets/143663952/267afeb1-6fe8-4b1e-9b07-c2810f576406)

Environment solved in 444 with an Average Score: 13.04

## Ideas for Future Work

Three different improvements to improve the algorithm are as follows:

Double DQN

Prioritized Experience Replay

Dueling DQN
