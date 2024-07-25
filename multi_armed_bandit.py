import numpy as np
import matplotlib.pyplot as plt

class MAB:
    def __init__(self, n_arms):
        self.n_arms = n_arms
        self.means = np.random.normal(0, 3, n_arms)
    
    def pull(self, arm):
        return np.random.normal(self.means[arm], 1)

class EpsilonGreedy:
    def __init__(self, n_arms, epsilon):
        self.n_arms = n_arms
        self.epsilon = epsilon
        self.q_values = np.zeros(n_arms)
        self.arm_counts = np.zeros(n_arms)
    
    def select_arm(self):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.n_arms)
        else:
            return np.argmax(self.q_values)
    
    def update(self, arm, reward):
        self.arm_counts[arm] += 1
        self.q_values[arm] += (reward - self.q_values[arm]) / self.arm_counts[arm]

class OptimisticGreedy:
    def __init__(self, n_arms, initial_value):
        self.n_arms = n_arms
        self.q_values = np.full(n_arms, initial_value)
        self.arm_counts = np.zeros(n_arms)
    
    def select_arm(self):
        return np.argmax(self.q_values)
    
    def update(self, arm, reward):
        self.arm_counts[arm] += 1
        self.q_values[arm] += (reward - self.q_values[arm]) / self.arm_counts[arm]

class UCB:
    def __init__(self, n_arms, c):
        self.n_arms = n_arms
        self.c = c
        self.q_values = np.zeros(n_arms)
        self.arm_counts = np.zeros(n_arms)
        self.total_counts = 0
    
    def select_arm(self):
        self.total_counts += 1
        ucb_values = self.q_values + self.c * np.sqrt(np.log(self.total_counts) / (self.arm_counts + 1e-5))
        return np.argmax(ucb_values)
    
    def update(self, arm, reward):
        self.arm_counts[arm] += 1
        self.q_values[arm] += (reward - self.q_values[arm]) / self.arm_counts[arm]

def simulate(mab, algorithm, steps):
    rewards = np.zeros(steps)
    for step in range(steps):
        arm = algorithm.select_arm()
        reward = mab.pull(arm)
        algorithm.update(arm, reward)
        rewards[step] = reward
    return rewards

def average_rewards(mab, algorithm, steps, runs=100):
    all_rewards = np.zeros((runs, steps))
    for run in range(runs):
        rewards = simulate(mab, algorithm, steps)
        all_rewards[run] = rewards
    average_rewards = np.mean(all_rewards, axis=0)
    return average_rewards

def main():
    n_arms = 10
    steps = 1000
    runs = 100

    epsilon_greedy = EpsilonGreedy(n_arms, epsilon=0.1)
    optimistic_greedy = OptimisticGreedy(n_arms, initial_value=5)
    ucb = UCB(n_arms, c=2)

    epsilon_greedy_rewards = average_rewards(MAB(n_arms), epsilon_greedy, steps, runs)
    optimistic_greedy_rewards = average_rewards(MAB(n_arms), optimistic_greedy, steps, runs)
    ucb_rewards = average_rewards(MAB(n_arms), ucb, steps, runs)

    # Plotting average rewards over time
    plt.figure(figsize=(10, 6))
    plt.plot(epsilon_greedy_rewards, label='Epsilon-Greedy (ε=0.1)')
    plt.plot(optimistic_greedy_rewards, label='Optimistic Greedy (Q1=5)')
    plt.plot(ucb_rewards, label='UCB (c=2)')
    plt.xlabel('Steps')
    plt.ylabel('Average Reward')
    plt.title('Average Reward over Time for Different Algorithms')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()