# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt

# MAB Implementation
class MAB:
    def __init__(self, n_arms):
        self.n_arms = n_arms
        self.means = np.random.normal(0, np.sqrt(3), n_arms)
        self.variances = np.ones(n_arms)

    def pull(self, arm):
        return np.random.normal(self.means[arm], np.sqrt(self.variances[arm]))

# Algorithms Implementation
class EpsilonGreedy:
    def __init__(self, n_arms, epsilon=0.1, initial_value=0):
        self.n_arms = n_arms
        self.epsilon = epsilon
        self.values = np.full(n_arms, initial_value)
        self.counts = np.zeros(n_arms)

    def select_arm(self):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.n_arms)
        else:
            return np.argmax(self.values)

    def update(self, arm, reward):
        self.counts[arm] += 1
        self.values[arm] += (reward - self.values[arm]) / self.counts[arm]

class OptimisticGreedy:
    def __init__(self, n_arms, initial_value=5):
        self.n_arms = n_arms
        self.values = np.full(n_arms, initial_value)
        self.counts = np.zeros(n_arms)

    def select_arm(self):
        return np.argmax(self.values)

    def update(self, arm, reward):
        self.counts[arm] += 1
        self.values[arm] += (reward - self.values[arm]) / self.counts[arm]

class UCB:
    def __init__(self, n_arms, c=2):
        self.n_arms = n_arms
        self.c = c
        self.values = np.zeros(n_arms)
        self.counts = np.zeros(n_arms)
        self.total_counts = 0

    def select_arm(self):
        if 0 in self.counts:
            return np.argmin(self.counts)
        else:
            ucb_values = self.values + self.c * np.sqrt(np.log(self.total_counts) / self.counts)
            return np.argmax(ucb_values)

    def update(self, arm, reward):
        self.total_counts += 1
        self.counts[arm] += 1
        self.values[arm] += (reward - self.values[arm]) / self.counts[arm]

# Experiment setup
def run_experiment(mab, algorithm, steps):
    rewards = np.zeros(steps)
    for step in range(steps):
        arm = algorithm.select_arm()
        reward = mab.pull(arm)
        algorithm.update(arm, reward)
        rewards[step] = reward
    return rewards

# Plotting results
def plot_results(rewards, labels, title):
    for i, reward in enumerate(rewards):
        plt.plot(np.cumsum(reward) / (np.arange(len(reward)) + 1), label=labels[i])
    plt.xlabel('Steps')
    plt.ylabel('Average Reward')
    plt.title(title)
    plt.legend()
    plt.show()

# Main
if __name__ == "__main__":
    n_arms = 10
    steps = 1000
    mab = MAB(n_arms)

    epsilon_greedy = EpsilonGreedy(n_arms, epsilon=0.1)
    optimistic_greedy = OptimisticGreedy(n_arms, initial_value=5)
    ucb = UCB(n_arms, c=2)

    epsilon_greedy_rewards = run_experiment(mab, epsilon_greedy, steps)
    optimistic_greedy_rewards = run_experiment(mab, optimistic_greedy, steps)
    ucb_rewards = run_experiment(mab, ucb, steps)

    plot_results([epsilon_greedy_rewards, optimistic_greedy_rewards, ucb_rewards],
                 ['Epsilon-Greedy', 'Optimistic Greedy', 'UCB'],
                 'Average Reward Over Time')
