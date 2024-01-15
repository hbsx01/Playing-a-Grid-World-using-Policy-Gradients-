# Playing a Grid-World using Policy Gradients

This project is an implementation of the REINFORCE algorithm, a Monte-Carlo policy-gradient control method, for training an agent to navigate a Grid-World environment, specifically a Short Corridor with Switched Actions. This work replicates the optimal policy convergence as demonstrated in Figure 13.1 from the seminal book "Reinforcement Learning: An Introduction" by Sutton and Barto.

## Project Overview

The Grid-World environment is a standard testbed for reinforcement learning algorithms. In this particular scenario, the agent is tasked with navigating through a short corridor with switched actions, which means that the usual action effects are reversed in certain states, adding complexity to the learning process.

The goal of this project was to train an agent that can consistently achieve an optimal state value function. The success of the agent is measured by the average total reward per episode (G0), which reached -10, indicating optimal navigation through the Grid-World's environment.

## Algorithm: REINFORCE

REINFORCE, or Monte-Carlo policy-gradient control, is a method that seeks to maximize the expected cumulative reward by adjusting the policy in the direction of higher expected returns. Unlike value-based methods, policy gradients directly model the policy and can thus handle high-dimensional action spaces and continuous action spaces.

## Results

The agent, after training, demonstrates a strong understanding of the environment and navigates the grid with efficiency. The average total reward per episode of -10 is a clear indicator of the agent's performance and its ability to replicate the optimal policy.
