# Nihal Ranchod -> 2427378
# Lisa Godwin -> 2437980

import numpy as np
import random
import matplotlib.pyplot as plt

class GridworldMDP:
    def __init__(self):
        self.grid_size = 7
        self.initial_state = (6, 0)
        self.goal_state = (0, 0)
        self.obstacles = [(2, i) for i in range(6)]
        self.actions = ['up', 'down', 'left', 'right']
        self.state = self.initial_state

    # Reset the environment to the initial state
    def reset(self):
        self.state = self.initial_state
        return self.state

    def step(self, action):
        if action not in self.actions:
            raise ValueError("Invalid action")

        x, y = self.state
        if action == 'up':
            x = max(x - 1, 0)
        elif action == 'down':
            x = min(x + 1, self.grid_size - 1)
        elif action == 'left':
            y = max(y - 1, 0)
        elif action == 'right':
            y = min(y + 1, self.grid_size - 1)

        new_state = (x, y)
        if new_state in self.obstacles:
            new_state = self.state

        reward = -1
        if new_state == self.goal_state:
            reward = 20

        self.state = new_state
        return new_state, reward

# Random Agent
def random_agent(env, steps=50):
    state = env.reset()
    total_reward = 0
    for _ in range(steps):
        action = random.choice(env.actions)
        state, reward = env.step(action)
        total_reward += reward
        if state == env.goal_state:
            break
    return total_reward

# Optimal Value Grid
optimal_value_function = np.array(    [
        [20, 19, 18, 17, 16, 15, 14],
        [19, 18, 17, 16, 15, 14, 13],
        [-1, -1, -1, -1, -1, -1, 12],
        [5, 6, 7, 8, 9, 10, 11],
        [4, 5, 6, 7, 8, 9, 10],
        [3, 4, 5, 6, 7, 8, 9],
        [2, 3, 4, 5, 6, 7, 8],
    ])

# Greedy Agent
def greedy_agent(env, optimal_value_grid, steps=50):
    state = env.reset()
    total_reward = 0
    for _ in range(steps):
        x, y = state
        best_action = None
        best_value = -float('inf')
        
        for action in env.actions:
            if action == 'up':
                new_x, new_y = max(x - 1, 0), y
            elif action == 'down':
                new_x, new_y = min(x + 1, env.grid_size - 1), y
            elif action == 'left':
                new_x, new_y = x, max(y - 1, 0)
            elif action == 'right':
                new_x, new_y = x, min(y + 1, env.grid_size - 1)
            
            if (new_x, new_y) not in env.obstacles and optimal_value_grid[new_x][new_y] > best_value:
                best_value = optimal_value_grid[new_x][new_y]
                best_action = action
        
        state, reward = env.step(best_action)
        total_reward += reward
        if state == env.goal_state:
            break
    return total_reward

def main():
   # Run experiments
    env = GridworldMDP()
    random_returns = [random_agent(env) for _ in range(20)]
    greedy_returns = [greedy_agent(env, optimal_value_function) for _ in range(20)]

    print(random_returns)
    print(greedy_returns)
    # Plot results
    plt.bar(['Random Agent', 'Greedy Agent'], [np.mean(random_returns), np.mean(greedy_returns)])
    plt.ylabel('Average Return')
    plt.show() 
    
if __name__ == '__main__':
    main()