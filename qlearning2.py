import numpy as np
import random
from mazegen import makeMaze

# Define the maze (5x5 for illustration)
maze = np.array(makeMaze(12))

# Maze size
n_rows, n_cols = maze.shape
n_states = n_rows * n_cols
n_actions = 4  # Up, Down, Left, Right

# Initialize Q-table
Q = np.zeros((n_states, n_actions))

# Q-learning parameters
learning_rate = 0.1
discount_factor = 0.9
epsilon = 0.1
n_episodes = 1000

# Define start and goal
start = (1, 1)  # Top-left corner
goal = (23, 23)  # Bottom-right corner

# Convert state to row, col
def to_row_col(state):
    return (state // n_cols, state % n_cols)

# Convert row, col to state
def to_state(row, col):
    return row * n_cols + col

# Check if the current state is a terminal state (goal)
def is_terminal_state(row, col):
    return row == goal[0] and col == goal[1]

# Get next state and reward
def step(state, action):
    row, col = to_row_col(state)

    if action == 0:  # Up
        row = max(row - 1, 0)
    elif action == 1:  # Down
        row = min(row + 1, n_rows - 1)
    elif action == 2:  # Left
        col = max(col - 1, 0)
    elif action == 3:  # Right
        col = min(col + 1, n_cols - 1)

    if maze[row, col] == 1:  # Hitting a wall
        reward = -1
    elif is_terminal_state(row, col):  # Goal reached
        reward = 10
    else:  # Normal move
        reward = -0.01

    return to_state(row, col), reward, is_terminal_state(row, col)

# Training the Q-learning model
for episode in range(n_episodes):
    state = to_state(*start)  # Start state
    done = False

    while not done:
        # Choose action (epsilon-greedy policy)
        if random.uniform(0, 1) < epsilon:
            action = random.randint(0, n_actions - 1)
        else:
            action = np.argmax(Q[state])

        next_state, reward, done = step(state, action)

        # Update Q-table
        Q[state, action] += learning_rate * (reward + discount_factor * np.max(Q[next_state]) - Q[state, action])

        state = next_state

# Function to find the path from start to goal using the Q-table
def solve_maze():
    state = to_state(*start)
    path = [to_row_col(state)]
    while not is_terminal_state(*path[-1]):
        action = np.argmax(Q[state])
        state, _, _ = step(state, action)
        path.append(to_row_col(state))
    return path

# Find the path
path = solve_maze()
print("Path from start to goal:", path)
