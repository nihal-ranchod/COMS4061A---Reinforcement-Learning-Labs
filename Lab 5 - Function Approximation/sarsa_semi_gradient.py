import numpy as np
import matplotlib.pyplot as plt
import gym
from value_function import ValueFunction

def sarsa_semi_gradient(
    env,
    num_episodes: int,
    discount_factor: float,
    epsilon: float,
    alpha: float,
) -> np.ndarray:
    num_actions = env.action_space.n # number of possible actions in the environment
    q = ValueFunction(alpha, num_actions) # Initialise value function approximator
    steps_per_episode = np.zeros(num_episodes, dtype=int)

    for episode in range(num_episodes):
        state, _ = env.reset()
        action = q.act(state, epsilon) # Select action using epsilon-greedy policy
        steps = 0

        while True:
            new_state, reward, done, truncated, _ = env.step(action)

            if done or truncated:
                q.update(reward, state, action)
                break

            new_action = q.act(new_state, epsilon)
            target = reward + discount_factor * q(new_state, new_action) # Compute target Q-value
            q.update(target, state, action)

            state = new_state
            action = new_action
            steps += 1

        steps_per_episode[episode] = steps

    return steps_per_episode, q

def plot_episode_lengths(steps_per_episode: np.ndarray, num_episodes: int) -> None:
    plt.figure(figsize=(10, 6))
    plt.plot(np.arange(num_episodes), steps_per_episode, label='Steps per Episode')
    plt.yscale('log')
    plt.xlabel('Episode')
    plt.ylabel('Steps per Episode (log scale)')
    plt.title('SARSA Semi-Gradient with Tile Coding on MountainCar')
    plt.legend()
    plt.grid(True)
    plt.savefig("Lab 5 - Function Approximation/sarsa_semi_gradient.png")
    plt.show()

def run_experiment(epsilon: float, alpha: float, num_episodes: int, average_runs: int) -> ValueFunction:
    discount_factor = 1.0
    env = gym.make("MountainCar-v0")
    average_steps_per_episode = np.zeros(num_episodes, dtype=float)

    for _ in range(average_runs):
        steps_per_episode, q = sarsa_semi_gradient(env, num_episodes, discount_factor, epsilon, alpha)
        average_steps_per_episode += steps_per_episode

    average_steps_per_episode /= average_runs
    plot_episode_lengths(average_steps_per_episode, num_episodes)
    
    return q

def show_trained_model(num_episodes: int, epsilon: float, alpha: float) -> None:
    discount = 1.0
    env = gym.make("MountainCar-v0")
    _, q = sarsa_semi_gradient(env, num_episodes, discount, epsilon, alpha)

    env = gym.make("MountainCar-v0", render_mode="human")

    policy = lambda state: np.argmax([q(state, action) for action in range(env.action_space.n)])

    state, _ = env.reset()
    env.render()
    action = policy(state)

    while True:
        new_state, reward, done, truncated, _ = env.step(action)

        if done or truncated:
            break

        new_action = policy(new_state)

        state = new_state
        action = new_action

def main() -> None:
    alpha = 0.1
    epsilon = 0.1
    num_episodes = 500
    average_runs = 100
    q = run_experiment(epsilon, alpha, num_episodes, average_runs)
    
    # Show the trained model
    show_trained_model(num_episodes, epsilon, alpha)

if __name__ == "__main__":
    main()