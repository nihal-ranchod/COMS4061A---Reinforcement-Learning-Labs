import gym
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import imageio

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
    """Run SARSA(λ) algorithm on the environment and return Q-values."""
    Q = np.zeros((num_states, num_actions))
    Q_values_per_episode = np.zeros((num_episodes, num_states, num_actions))
    
    for episode in range(num_episodes):
        state = env.reset()
        if isinstance(state, tuple):
            state = state[0]
        action = epsilon_greedy_policy(Q, state, epsilon)
        E = np.zeros_like(Q)  # Eligibility traces

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

            if done:
                break

        # Store Q-values at the end of this episode
        Q_values_per_episode[episode] = Q.copy()

    return Q_values_per_episode

def save_heatmaps_for_animation(Q_values_per_episode, lambda_values, episode):
    """Save side-by-side heatmaps for different lambda values at each episode."""
    plt.figure(figsize=(15, 4))  # Adjust figsize as needed
    
    for i, lambda_ in enumerate(lambda_values):
        lambda_str = str(lambda_).replace('.', '_')
        ax = plt.subplot(1, len(lambda_values), i + 1)
        
        averaged_value_function = np.max(Q_values_per_episode[lambda_][episode], axis=1).reshape(grid_shape)
        
        sns.heatmap(
            averaged_value_function, 
            cmap='mako', 
            annot=False, 
            fmt=".1f", 
            cbar=True, 
            square=True, 
            xticklabels=False, 
            yticklabels=False, 
            ax=ax
        )
        
        ax.set_title(f'λ={lambda_str}')

    plt.suptitle(f'Episode {episode + 1}', fontsize=16)
    
    folder = "Lab 4 - Model Free Learning/animations"
    os.makedirs(folder, exist_ok=True)
    plt.savefig(f"{folder}/episode_{episode + 1}.png")
    plt.close()

def generate_animation():
    # Experiment parameters
    num_episodes = 200
    lambda_values = [0, 0.3, 0.5]
    alpha = 0.5
    epsilon = 0.1

    # Store Q-values per λ for all episodes
    Q_values_per_episode = {lambda_: sarsa_lambda(env, lambda_, num_episodes, alpha, epsilon) for lambda_ in lambda_values}

    # Save heatmaps for each episode
    for episode in range(num_episodes):
        save_heatmaps_for_animation(Q_values_per_episode, lambda_values, episode)

    # Create the animation
    images = []
    folder = "Lab 4 - Model Free Learning/animations"
    for episode in range(1, num_episodes + 1):
        images.append(imageio.imread(f"{folder}/episode_{episode}.png"))
    
    imageio.mimsave('sarsa_lambda_animation.gif', images, duration=0.5)  # Adjust duration as needed

if __name__ == '__main__':
    generate_animation()
