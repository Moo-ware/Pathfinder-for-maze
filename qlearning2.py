import numpy as np
import random

class MazeSolver:
    def __init__(self, maze, start, goal, learning_rate=0.1, discount_factor=0.9, epsilon=0.1, n_episodes=1000):
        self.maze = maze
        self.start = start
        self.goal = goal
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.n_episodes = n_episodes

        # Maze dimensions
        self.n_rows, self.n_cols = maze.shape
        self.n_states = self.n_rows * self.n_cols
        self.n_actions = 4  # Up, Down, Left, Right

        # Initialize Q-table
        self.Q = np.zeros((self.n_states, self.n_actions))

    def to_row_col(self, state):
        return (state // self.n_cols, state % self.n_cols)

    def to_state(self, row, col):
        return row * self.n_cols + col

    def is_terminal_state(self, row, col):
        return row == self.goal[0] and col == self.goal[1]

    def step(self, state, action):
        row, col = self.to_row_col(state)

        # Action logic
        if action == 0: row = max(row - 1, 0)
        elif action == 1: row = min(row + 1, self.n_rows - 1)
        elif action == 2: col = max(col - 1, 0)
        elif action == 3: col = min(col + 1, self.n_cols - 1)

        # Reward logic
        if self.maze[row, col] == 1: reward = -1
        elif self.is_terminal_state(row, col): reward = 10
        else: reward = -0.01

        return self.to_state(row, col), reward, self.is_terminal_state(row, col)

    def train(self):
        for episode in range(self.n_episodes):
            state = self.to_state(*self.start)
            done = False

            while not done:
                if random.uniform(0, 1) < self.epsilon:
                    action = random.randint(0, self.n_actions - 1)
                else:
                    action = np.argmax(self.Q[state])

                next_state, reward, done = self.step(state, action)

                # Q-table update
                self.Q[state, action] += self.learning_rate * (reward + self.discount_factor * np.max(self.Q[next_state]) - self.Q[state, action])

                state = next_state

    def solve(self):
        self.train()  # Ensure the model is trained

        state = self.to_state(*self.start)
        path = [self.to_row_col(state)]
        while not self.is_terminal_state(*path[-1]):
            action = np.argmax(self.Q[state])
            state, _, _ = self.step(state, action)
            path.append(self.to_row_col(state))

        return path
