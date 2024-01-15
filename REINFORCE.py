import numpy as np

class ShortCorridorWithSwitchedActionsEnv:
    def __init__(self):
        self.reset()
        self.num_actions = 2
        self.num_states = 3

    def reset(self):
        self.state = 0
        return self.state

    def step(self, action):
        assert action in [0, 1]

        if self.state == 0:
            if action == 1:
                new_state = self.state + 1
            else:
                new_state = self.state  # Bumped into the left wall
        elif self.state == 1:
            if action == 1:
                new_state = self.state - 1  # Switched action
            else:
                new_state = self.state + 1  # Switched action
        elif self.state == 2:
            if action == 1:
                new_state = self.state + 1
            else:
                new_state = self.state - 1
        else:
            # Terminal state, stay there
            new_state = self.state

        # Always receive a reward of -1 until the terminal state is reached
        reward = -1
        is_terminal = new_state == 3

        self.state = new_state
        return new_state, reward, is_terminal

class REINFORCE:
    def __init__(self, num_actions, num_states, learning_rate=0.01):
        self.num_actions = num_actions
        self.num_states = num_states
        self.theta = np.zeros((num_states, num_actions))  
        self.alpha = learning_rate
        self.gamma = 0.8  # Discount factor
        
    def policy(self, state):
        #Compute the policy for a given state using softmax.
        preferences = self.theta.T.dot(state)
        max_preference = np.max(preferences)
        exp_preferences = np.exp(preferences - max_preference)
        softmax_probabilities = exp_preferences / np.sum(exp_preferences)
        return softmax_probabilities

    def select_action(self, state):
        #Select an action according to the softmax policy.
        probabilities = self.policy(state)
        return np.random.choice(self.num_actions, p=probabilities)

    def update_policy(self, episode):
        G = 0
        for state, action, reward in reversed(episode):
            G = reward + self.gamma * G
            softmax_probabilities = self.policy(state)
            d_softmax = softmax_probabilities.copy()
            d_softmax[action] -= 1  # Subtract 1 from the action taken
            grad_ln_policy = np.dot(state[:, np.newaxis], d_softmax[np.newaxis, :])  # Gradient of log policy
            self.theta -= self.alpha * grad_ln_policy * G  # Update theta
    
def one_hot(state, num_states=3):
    vec = np.zeros(num_states)
    vec[state] = 1
    return vec

def run_experiment(alphas, num_episodes):
    rewards_per_alpha = {alpha: [] for alpha in alphas}
    
    for alpha in alphas:
        average_rewards = []
        env = ShortCorridorWithSwitchedActionsEnv()
        policy = REINFORCE(env.num_actions, env.num_states, learning_rate=alpha)
        total_rewards = []
        for episode in range(num_episodes):
            print("episode" , episode)
            state = env.reset()
            episode_history = []
            total_reward = 0
            for _ in range(100):  # Max 1000 steps to avoid infinite loops
                one_hot_state = one_hot(state)
                action = policy.select_action(one_hot_state)
                new_state, reward, done = env.step(action)
                total_reward += reward
                episode_history.append((one_hot_state, action, reward))  # Store one_hot_state
                env.state = new_state
                state = new_state
                print((one_hot_state, action, reward))
                print(total_reward)
                if done:
                    print("game finished")
                    break
            policy.update_policy(episode_history)
            total_rewards.append(total_reward)
            average_rewards.append(np.mean(total_rewards))
        rewards_per_alpha[alpha] = average_rewards

    plt.figure(figsize=(12, 8))
    for alpha, rewards in rewards_per_alpha.items():
        plt.plot(rewards, label=f'alpha={alpha}')
    
    plt.xlabel('Episodes')
    plt.ylabel('Total reward per episode')
    plt.legend()
    plt.title('REINFORCE on Short Corridor with Switched Actions')
    plt.show()

alphas = [2**-13, 2**-14, 2**-12]
num_episodes = 1000
run_experiment(alphas, num_episodes)