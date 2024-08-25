import gym
import numpy as np
import matplotlib.pyplot as plt

# Initialize the CliffWalking environment
env = gym.make('CliffWalking-v0')
num_states = env.observation_space.n
num_actions = env.action_space.n

def epsilon_greedy_policy(Q, state, epsilon):
    """Select action using ε-greedy policy."""
    if np.random.rand() < epsilon:
        return np.random.choice(num_actions)
    else:
        return np.argmax(Q[state])

def sarsa_lambda(env, lambda_, num_episodes, alpha, epsilon):
    """Run SARSA(λ) algorithm on the environment and return Q-values and returns."""
    Q = np.zeros((num_states, num_actions))
    returns = []
    Q_values_per_episode = np.zeros((num_episodes, num_states, num_actions))
    
    for episode in range(num_episodes):
        state = env.reset()
        if isinstance(state, tuple):
            state = state[0]
        action = epsilon_greedy_policy(Q, state, epsilon)
        E = np.zeros_like(Q)  # Eligibility traces
        total_return = 0

        while True:
            next_state, reward, done = env.step(action)[:3]
            if isinstance(next_state, tuple):
                next_state = next_state[0]
            next_action = epsilon_greedy_policy(Q, next_state, epsilon)
            delta = reward + Q[next_state, next_action] - Q[state, action]
            E[state, action] += 1

            # Update Q values and eligibility traces
            Q += alpha * delta * E
            E *= lambda_

            state, action = next_state, next_action
            total_return += reward

            if done:
                break

        # Store Q-values at the end of this episode
        Q_values_per_episode[episode] = Q.copy()
        returns.append(total_return)

    return Q_values_per_episode, returns

def main():
    # Experiment parameters
    num_runs = 100
    num_episodes = 200
    lambda_values = [0, 0.3, 0.5]
    alpha = 0.5
    epsilon = 0.1

    plt.figure(figsize=(12, 8))

    for lambda_ in lambda_values:
        all_Q_values = np.zeros((num_episodes, num_states, num_actions))

        all_returns = []

        for run in range(num_runs):
            Q_values_per_episode, returns = sarsa_lambda(env, lambda_, num_episodes, alpha, epsilon)
            all_Q_values += Q_values_per_episode
            all_returns.append(returns)

        # Calculate average return and standard deviation over all runs
        avg_returns = np.mean(all_returns, axis=0)
        std_returns = np.std(all_returns, axis=0)

        # Plotting the average return with shading for standard deviation
        plt.plot(avg_returns, label=f'λ = {lambda_}')
        plt.fill_between(range(num_episodes), avg_returns - std_returns, avg_returns + std_returns, alpha=0.2)

    # Set y-axis range based on expected returns in CliffWalking
    plt.ylim(-500, 10)  # Adjust as needed

    plt.xlabel('Episodes')
    plt.ylabel('Average Return')
    plt.title('SARSA(λ) Performance on CliffWalking')
    plt.legend(loc='lower right') 
    plt.grid(True)
    plt.savefig("Lab 4 - Model Free Learning/sarsa_lambda_average_return.png")
    plt.show()

if __name__ == '__main__':
    main()
