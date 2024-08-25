import gym
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Initialize the CliffWalking environment
env = gym.make('CliffWalking-v0')
num_states = env.observation_space.n
num_actions = env.action_space.n
grid_shape = (4, 12)  # For plotting the grid

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

def save_averaged_heatmap(average_Q_values, episode, lambda_str):
    averaged_value_function = np.max(average_Q_values, axis=1).reshape(grid_shape)
    plt.figure(figsize=(8, 6))
    
    sns.heatmap(
        averaged_value_function, 
        cmap=sns.cubehelix_palette(as_cmap=True), 
        annot=False, 
        fmt=".1f", 
        cbar=True, 
        square=True, 
        xticklabels=False, 
        yticklabels=False
    )
    
    plt.title(f'Average Value Function Heatmap\nλ={lambda_str}, Episode {episode + 1}')
    
    folder = f"Lab 4 - Model Free Learning/heatmaps/averaged_lambda_{lambda_str}"
    os.makedirs(folder, exist_ok=True)
    plt.savefig(f"{folder}/episode_{episode + 1}.png")
    plt.close()

def main():
    # Experiment parameters
    num_runs = 100
    num_episodes = 200
    lambda_values = [0, 0.3, 0.5]
    alpha = 0.5
    epsilon = 0.1

    plt.figure(figsize=(12, 8))

    for lambda_ in lambda_values:
        lambda_str = str(lambda_).replace('.', '_')  # For folder naming
        all_Q_values = np.zeros((num_episodes, num_states, num_actions))

        all_returns = []

        for run in range(num_runs):
            Q_values_per_episode, returns = sarsa_lambda(env, lambda_, num_episodes, alpha, epsilon)
            all_Q_values += Q_values_per_episode
            all_returns.append(returns)
        
        # Average the Q-values across all runs
        averaged_Q_values = all_Q_values / num_runs

        # Save the heatmaps for each episode
        for episode in range(num_episodes):
            save_averaged_heatmap(averaged_Q_values[episode], episode, lambda_str)

        # Calculate average return and standard deviation over all runs
        avg_returns = np.mean(all_returns, axis=0)
        std_returns = np.std(all_returns, axis=0)

        # Plotting the average return with shading for standard deviation
        plt.plot(avg_returns, label=f'λ = {lambda_}')
        plt.fill_between(range(num_episodes), avg_returns - std_returns, avg_returns + std_returns, alpha=0.2)

    plt.xlabel('Episodes')
    plt.ylabel('Average Return')
    plt.title('SARSA(λ) Performance on CliffWalking')
    plt.legend(loc=4)
    plt.grid(True)
    plt.savefig('sarsa_lambda_average_return.png')
    plt.show()

if __name__ == '__main__':
    main()
