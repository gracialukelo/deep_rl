from typing import Any, Tuple

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt


class Agent:
    def __init__(self, env: gym.Env) -> None:
        self.env = env
        self.observations: int = self.env.observation_space.n
        self.actions: int = self.env.action_space.n
        self.gamma = 0.95  # Discount factor
        self.alpha = 0.20  # Learning rate
        self.state = self.env.reset()[0]
        self.S = range(self.observations)
        self.A = range(self.actions)
        self.q_values = {s: {a: 0.0 for a in self.A} for s in self.S}

    def get_action(self, state: Any) -> int:
        q_values = list(self.q_values[state].values())
        return np.argmax(q_values).astype(int)

    def get_random_action(self) -> int:
        return self.env.action_space.sample()

    def get_v_values(self, state: Any) -> float:
        q_values = list(self.q_values[state].values())
        v_value: float = np.max(q_values).astype(float)
        return v_value

    def get_sample(self) -> Tuple[int, int, float, int]:
        old_state = self.state
        action = self.get_random_action()
        new_state, reward, done, _, _ = self.env.step(action)
        if done:
            self.state = self.env.reset()[0]
        else:
            self.state = new_state
        return old_state, action, reward, new_state

    def compute_q_values(self, state: int, action: int, reward: float, state_next: int) -> None:
        v_value_next = self.get_v_values(state_next)
        update_q_value = reward + self.gamma * v_value_next
        q_value_action = self.q_values[state][action]
        new_q_value = (1.0 - self.alpha) * q_value_action + self.alpha * update_q_value
        self.q_values[state][action] = new_q_value

    def train(self, num_iterations: int) -> None:
        best_reward_mean = -np.inf
        for iteration in range(num_iterations):
            state, action, reward, next_state = self.get_sample()
            self.compute_q_values(state, action, reward, next_state)
            reward_mean = self.play(num_episodes=20, render=False)
            if iteration % 250 == 0:
                print(f"Iteration: {iteration}, Current Reward Mean: {reward_mean}")
            if reward_mean > best_reward_mean:
                print(
                    f"Iteration {iteration}: "
                    f"Old best_reward_mean: {best_reward_mean}, "
                    f"New best_reward_mean: {reward_mean}"
                )
                best_reward_mean = reward_mean

    def play(self, num_episodes: int, render: bool = True) -> float:
        reward_sum = 0.0
        for episode in range(num_episodes):
            state = self.env.reset()[0]
            total_reward = 0.0
            while True:
                action = self.get_action(state)
                if render:
                    self.env.render()
                state, reward, done, _, _ = self.env.step(action)
                total_reward += reward
                if done:
                    print(f"Episode: {episode + 1}, Total Reward: {total_reward}")
                    reward_sum += total_reward
                    break
        return reward_sum / num_episodes


def main() -> None:
    env = gym.make("Taxi-v3")
    agent = Agent(env)
    agent.train(num_iterations=5000)
    agent.play(num_episodes=5, render=True)


if __name__ == "__main__":
    main()
