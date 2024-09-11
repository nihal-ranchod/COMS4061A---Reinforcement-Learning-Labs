import numpy as np
import matplotlib.pyplot as plt
import gym
from value_function import ValueFunction
from gym.wrappers import RecordVideo

def sarsa_semi_gradient(
    env,
    num_episodes: int,
    discount_factor: float,
    epsilon: float,
    alpha: float,
) -> np.ndarray:
    num_actions = env.action_space.n
    q = ValueFunction(alpha, num_actions)
    steps_per_episode = np.zeros(num_episodes, dtype=int)

    for episode in range(num_episodes):
        state, _ = env.reset()
        action = q.act(state, epsilon)
        steps = 0

        while True:
            new_state, reward, done, truncated, _ = env.step(action)

            if done or truncated:
                q.update(reward, state, action)
                break

            new_action = q.act(new_state, epsilon)
            target = reward + discount_factor * q(new_state, new_action)
            q.update(target, state, action)

            state = new_state
            action = new_action
            steps += 1

        steps_per_episode[episode] = steps

    return steps_per_episode, q

def plot_episode_lengths(steps_per_episode: np.ndarray, num_episodes: int, save_path: str) -> None:
    plt.figure(figsize=(10, 6))
    plt.plot(np.arange(num_episodes), steps_per_episode, label='Steps per Episode')
    plt.yscale('log')
    plt.xlabel('Episode')
    plt.ylabel('Steps per Episode (log scale)')
    plt.title('SARSA Semi-Gradient with Tile Coding on MountainCar')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)  # Save the plot to a file
    plt.show()

def run_experiment(epsilon: float, alpha: float, num_episodes: int, average_runs: int) -> ValueFunction:
    discount_factor = 1.0
    env = gym.make("MountainCar-v0")
    average_steps_per_episode = np.zeros(num_episodes, dtype=float)

    for _ in range(average_runs):
        steps_per_episode, q = sarsa_semi_gradient(env, num_episodes, discount_factor, epsilon, alpha)
        average_steps_per_episode += steps_per_episode

    np.save('sarsa_semi_gradient.npy', q.weights)  # Save the Q-values

    average_steps_per_episode /= average_runs
    
    # Save the steps per episode to a file
    np.save('steps_per_episode.npy', average_steps_per_episode)
    
    # Plot and save the plot
    plot_episode_lengths(average_steps_per_episode, num_episodes, 'sarsa_plot.png')
    
    return q

def render_policy(env, q: ValueFunction, video_path: str) -> None:
    env = RecordVideo(env, video_path, force=True)
    state, _ = env.reset()
    done = False
    while not done:
        action = q.act(state, epsilon=0.0)  # Use the learned policy without exploration
        state, _, done, _, _ = env.step(action)
        env.render()
    env.close()

def load_and_render_policy(env, q_weights_path: str, video_path: str) -> None:
    num_actions = env.action_space.n
    q = ValueFunction(alpha=0.1, num_actions=num_actions)
    q.weights = np.load(q_weights_path)  # Load the saved Q-values
    render_policy(env, q, video_path)

def main() -> None:
    alpha = 0.1
    epsilon = 0.1
    num_episodes = 500
    average_runs = 100
    q = run_experiment(epsilon, alpha, num_episodes, average_runs)
    
    # Render the policy and create a video
    env = gym.make("MountainCar-v0")
    load_and_render_policy(env, 'sarsa_semi_gradient.npy', './video')

if __name__ == "__main__":
    main()