# Learning Algorithm

This project was solved using a Deep Q-Network learning algorithm. Deep Q-Networks involve alternating between sampling the environment and learning.

The sampling the environment step involves taking epsilon greedy actions A from state S using a local version of the neural network to get the current estimate of q. The environment returns back a reward R and next state S'. The (S,A,R,S') tuples are stored in replay memory D. 

The learning step involves taking random minibatch of tuples from D and learning from the batch using gradient descent to update the weights of a local neural network. The algorithm maintains two versions of the neural network architecture, a target version and a local version. The Q_targets_next are calculated by passing the minibatch of next states, S', to the target network and selecting the maximum q value over all possible actions for each tuple in the minibatch. Then Q_targets are computed for each tuple in the minibatch by adding the rewards to gamma * Q_targets_next * (1 - dones), where dones is 0 or 1 indicating whether a terminal state has been reached. Q_expected is calculated by passing the states, S, for each tuple in the minibatch through the local network and taking the action A. Then the MSE Loss between Q_expected and Q_targets is calculated, a gradient descent step occurs to update the local weights, and the target weights are updated to be TAU*(local weights) + (1 - TAU)*(target weights) 

## The hyperparameters chosen are as follows
- n_episodes=2000 (number of episodes for which to train)
- max_t=1000 (maximum number of time steps per episode)
- eps_start=1.0 (max value of epsilon)
- eps_end=0.01 (min value of epsilon)
- eps_decay=0.995 (value of (epsilon at episode n+1)/(epsilon at episode n))
- BUFFER_SIZE = int(1e6)  (size of the replay buffer)
- BATCH_SIZE = 64         (number of tuples to sample from memory during a learning step)
- GAMMA = 0.99            (discount factor of reward at t+1 to reward at t)
- TAU = 1e-3              (soft update of target parameters based on local parameters)
- LR = 5e-4               (learning rate) 
- UPDATE_EVERY = 4        (how many time steps to wait before updating the network)

## Model architectures for any neural networks

Both the target and local neural networks have the following architecture:
- A first fully connected layer of input size state_size=37 to output size fc1_units=64 followed by a relu activation function
- A second fully connected layer of input size fc1_units=64 to output size fc2_units=64 followed by a relu activation function
- A third fully connected layer of input size fc2_units=64 fo output size action_size=4

## Plot of Rewards

As shown in this plot, the trained agent is able to receive an average reward (over 100 episodes) of at least +13. 

![AverageScoreLast100Episodes](https://github.com/DataScience087/Project1-Navigation/assets/143663952/267afeb1-6fe8-4b1e-9b07-c2810f576406)

Environment solved in 444 episodes with an Average Score: 13.04

## Ideas for Future Work

Three different improvements to improve the algorithm are as follows:

Double DQN: This improvement helps solve the issue of overestimation of Q-values that Q-learning is prone to. The TD target, in particular the max operation associated with it, are prone to mistakes especially in early stages. Double Q-learning can make estimation of the TD target more robust by selection of the best action using one set of parameters but evaluation of the q function using a different set of parameters.

Prioritized Experience Replay: This improvement helps solve the issue that some tuples in the replay memory, which could be very important to learning, may occur very infrequently in the replay memory. This method assigns priorities to each tuple according to the size of their associated TD error, and then it samples tuples from the replay memory with a probability equal to the tuple's priority divided by the sum of all priorities. Some improvements to the described priortied experience replay exist as well.  

Dueling DQN: This improvement attempts to estimate Q(s,a) with a state value function V(s) plus an advantage for each A(s,a). The Dueling Network has two streams, one which estimates the V(s) and one which estimates A(s,a).
