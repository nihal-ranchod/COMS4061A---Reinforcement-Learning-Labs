import numpy as np

# Define gridworld parameters
grid_size = 7
goal_state = (0, 0)
initial_state = (6, 0)
obstacles = [(2, i) for i in range(grid_size - 1)]  # Obstacles in the 3rd row except the last column
reward_goal = 20
reward_step = -1
gamma = 1  # No discounting

# Initialize value function
V = np.zeros((grid_size, grid_size))

# Define actions (up, down, left, right)
actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

# Define the transition and reward functions
def transition(state, action):
    new_state = (state[0] + action[0], state[1] + action[1])
    if new_state[0] < 0 or new_state[0] >= grid_size or new_state[1] < 0 or new_state[1] >= grid_size:
        return state, reward_step  # Out of bounds, stay in the same state
    if new_state in obstacles:
        return state, reward_step  # Obstacle, stay in the same state
    if new_state == goal_state:
        return new_state, reward_goal  # Goal state
    return new_state, reward_step  # Valid move

# Value Iteration
def value_iteration(V, theta=1e-6):
    while True:
        delta = 0
        for row in range(grid_size):
            for col in range(grid_size):
                state = (row, col)
                if state == goal_state:
                    continue  # Skip the goal state
                v = V[state]
                V[state] = max(transition(state, a)[1] + gamma * V[transition(state, a)[0]] for a in actions)
                delta = max(delta, abs(v - V[state]))
        if delta < theta:
            break
    return V

# Compute the optimal value function
optimal_V = value_iteration(V)
print("Optimal Value Function:")
print(optimal_V)
