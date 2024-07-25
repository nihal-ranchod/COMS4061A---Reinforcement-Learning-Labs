import numpy as np
import matplotlib.pyplot as plt

class Bandit:
    def __init__(self, k):
        self.k = k
        self.means = np.random.normal(0, np.sqrt(3), k)
    
    def pull(self, arm):
        return np.random.normal(self.means[arm], 1)

def epsilon_greedy(bandit, epsilon, steps):
    k = bandit.k
    Q = np.zeros(k)
    N = np.zeros(k)
    rewards = np.zeros(steps)
    
    for t in range(steps):
        if np.random.rand() < epsilon:
            arm = np.random.choice(k)
        else:
            arm = np.argmax(Q)
        
        reward = bandit.pull(arm)
        N[arm] += 1
        Q[arm] += (reward - Q[arm]) / N[arm]
        rewards[t] = reward
    
    return rewards

def greedy_optimistic(bandit, Q1, steps):
    k = bandit.k
    Q = np.ones(k) * Q1
    N = np.zeros(k)
    rewards = np.zeros(steps)
    
    for t in range(steps):
        arm = np.argmax(Q)
        reward = bandit.pull(arm)
        N[arm] += 1
        Q[arm] += (reward - Q[arm]) / N[arm]
        rewards[t] = reward
    
    return rewards

def ucb(bandit, c, steps):
    k = bandit.k
    Q = np.zeros(k)
    N = np.zeros(k)
    rewards = np.zeros(steps)
    
    for t in range(steps):
        if t < k:
            arm = t
        else:
            ucb_values = Q + c * np.sqrt(np.log(t + 1) / (N + 1e-5))
            arm = np.argmax(ucb_values)
        
        reward = bandit.pull(arm)
        N[arm] += 1
        Q[arm] += (reward - Q[arm]) / N[arm]
        rewards[t] = reward
    
    return rewards

def run_simulation(algorithm, bandit, param, steps, runs):
    all_rewards = np.zeros((runs, steps))
    for run in range(runs):
        rewards = algorithm(bandit, param, steps)
        all_rewards[run] = rewards
    return np.mean(all_rewards, axis=0)

# Parameters
k = 10
steps = 1000
runs = 100

# Initialize bandit
bandit = Bandit(k)

# Run simulations
epsilon_rewards = run_simulation(epsilon_greedy, bandit, 0.1, steps, runs)
optimistic_rewards = run_simulation(greedy_optimistic, bandit, 5, steps, runs)
ucb_rewards = run_simulation(ucb, bandit, 2, steps, runs)

# Plot results
plt.figure(figsize=(10, 6))
plt.plot(epsilon_rewards, label='𝜖-greedy (𝜖=0.1)')
plt.plot(optimistic_rewards, label='Greedy (Q1=5)')
plt.plot(ucb_rewards, label='UCB (c=2)')
plt.xlabel('Steps')
plt.ylabel('Average Reward')
plt.legend()
plt.title('Average Reward over Time')
plt.show()