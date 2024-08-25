import gym
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import imageio
from sarsa_lambda import sarsa_lambda

# Initialize the CliffWalking environment
env = gym.make('CliffWalking-v0')
grid_shape = (4, 12)  

def save_heatmaps_for_animation(Q_values_per_episode, lambda_values, episode):
    """Save side-by-side heatmaps for different lambda values at each episode."""
    plt.figure(figsize=(15, 4))  # Adjust figsize as needed
    
    for i, lambda_ in enumerate(lambda_values):
        ax = plt.subplot(1, len(lambda_values), i + 1)
        
        averaged_value_function = np.max(Q_values_per_episode[lambda_][episode], axis=1).reshape(grid_shape)
        
        sns.heatmap(
            averaged_value_function, 
            cmap='magma', 
            annot=False, 
            fmt=".1f", 
            cbar=True, 
            square=True, 
            xticklabels=False, 
            yticklabels=False, 
            ax=ax
        )
        
        ax.set_title(f'λ={lambda_}')

    plt.suptitle(f'Episode {episode + 1}', fontsize=16)
    
    folder = "Lab 4 - Model Free Learning/animations"
    os.makedirs(folder, exist_ok=True)
    plt.savefig(f"{folder}/episode_{episode + 1}.png")
    plt.close()

def generate_animation(num_episodes, lambda_values, alpha, epsilon):
    # Experiment parameters
    num_episodes = 200
    lambda_values = [0, 0.3, 0.5]
    alpha = 0.5
    epsilon = 0.1

    # Store Q-values per λ for all episodes
    Q_values_per_episode = {lambda_: sarsa_lambda(env, lambda_, num_episodes, alpha, epsilon)[0] for lambda_ in lambda_values}

    # Save heatmaps for each episode
    for episode in range(num_episodes):
        save_heatmaps_for_animation(Q_values_per_episode, lambda_values, episode)

    # Create the animation
    images = []
    folder = "Lab 4 - Model Free Learning/animations"
    for episode in range(1, num_episodes + 1):
        images.append(imageio.imread(f"{folder}/episode_{episode}.png"))
    
    imageio.mimsave("Lab 4 - Model Free Learning/sarsa_lambda_animation.gif", images, duration=0.5)  # Adjust duration as needed

def main():
    num_episodes = 200
    lambda_values = [0, 0.3, 0.5]
    alpha = 0.5
    epsilon = 0.1

    generate_animation(num_episodes, lambda_values, alpha, epsilon)

if __name__ == '__main__':
    main()