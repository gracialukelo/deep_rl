import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from typing import List

# Initialize the environment
env = gym.make("FrozenLake-v1", desc=None, map_name="4x4", is_slippery=True, render_mode="rgb_array")

# Parameters for Value Iteration
gamma: float = 0.99    # Discount factor
theta: float = 1e-6    # Convergence threshold

def value_iteration(env: gym.Env, gamma: float, epsilon: float) -> np.ndarray:
    value_table: np.ndarray = np.zeros(env.observation_space.n)

    while True:
        delta: float = 0  # Delta to check for convergence

        for state in range(env.observation_space.n):
            v_old: float = value_table[state]

            action_values: List[float] = []
            for action in range(env.action_space.n):
                action_value: float = sum([prob * (reward + gamma * value_table[next_state]) 
                                           for prob, next_state, reward, _ in env.P[state][action]])
                action_values.append(action_value)

            best_action_value: float = max(action_values)
            value_table[state] = best_action_value
            delta = max(delta, abs(v_old - best_action_value))

        if delta < epsilon:
            break

    return value_table

def extract_policy(env: gym.Env, value_table: np.ndarray, gamma: float) -> np.ndarray:
    policy: np.ndarray = np.zeros(env.observation_space.n)

    for state in range(env.observation_space.n):
        action_values: List[float] = []
        for action in range(env.action_space.n):
            action_value: float = sum([prob * (reward + gamma * value_table[next_state])
                                       for prob, next_state, reward, _ in env.P[state][action]])
            action_values.append(action_value)
        policy[state] = np.argmax(action_values)

    return policy

def play_game(env: gym.Env, policy: np.ndarray) -> int:
    state, _ = env.reset()
    total_reward: int = 0
    frames = []  # To store frames for rendering

    while True:
        action: int = int(policy[state])
        state, reward, done, _, _ = env.step(action)
        total_reward += reward
        frames.append(env.render())  # Store the frame

        if done:
            break

    # Display all frames as an animation
    for frame in frames:
        plt.imshow(frame)
        plt.axis('off')
        plt.show()

    return total_reward

if __name__ == "__main__":
    value_table: np.ndarray = value_iteration(env, gamma, theta)
    policy: np.ndarray = extract_policy(env, value_table, gamma)
    print("Optimal Value Table:")
    print(value_table.reshape((4, 4)))

    print("\nOptimal Policy:")
    print(policy.reshape((4, 4)))

    actions: List[str] = ["Left", "Down", "Right", "Up"]
    policy_actions: np.ndarray = np.array([actions[int(action)] for action in policy])
    print("\nOptimal Policy (Actions):")
    print(policy_actions.reshape((4, 4)))

    print("\nStarting the game...")
    total_reward: int = play_game(env, policy)
    print(f"Total Reward: {total_reward}")
