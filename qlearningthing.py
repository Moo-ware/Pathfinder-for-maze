import numpy as np

class QLearningMazeSolver:
    def __init__(self, maze, learning_rate=0.1, discount_factor=0.9, exploration_rate=0.1, epochs=1000):
        self.maze = maze
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.epochs = epochs
        self.actions = [(0, 1), (0, -1), (1, 0), (-1, 0)]  # Right, Left, Down, Up
        self.q_table = np.zeros((len(maze), len(maze[0]), len(self.actions)))
        
    def choose_action(self, state):
        if np.random.rand() < self.exploration_rate:
            return np.random.randint(len(self.actions))  # Explore action space
        else:
            return np.argmax(self.q_table[state[0], state[1]])  # Exploit learned values
    
    def update_q_table(self, state, action, reward, next_state):
        best_next_action = np.argmax(self.q_table[next_state[0], next_state[1]])
        current_q_value = self.q_table[state[0], state[1], action]
        new_q_value = current_q_value + self.learning_rate * (reward + 
                            self.discount_factor * self.q_table[next_state[0], next_state[1], best_next_action] - current_q_value)
        self.q_table[state[0], state[1], action] = new_q_value
    
    def train(self, start):
        for _ in range(self.epochs):
            state = start
            done = False
            while not done:
                action = self.choose_action(state)
                next_state, reward, done = self.step(state, action)
                self.update_q_table(state, action, reward, next_state)
                state = next_state
    
    def step(self, state, action):
        next_state = (state[0] + self.actions[action][0], state[1] + self.actions[action][1])
        if self.is_valid_move(next_state):
            if self.maze[next_state[0]][next_state[1]] == 'G':
                return next_state, 1, True  # Reached goal
            return next_state, 0, False  # Valid move
        return state, -1, True  # Invalid move
    
    def is_valid_move(self, state):
        rows = len(self.maze)
        cols = len(self.maze[0])
        return 0 <= state[0] < rows and 0 <= state[1] < cols and self.maze[state[0]][state[1]] != '#'
    
    def find_path(self, start):
        path = [start]
        state = start
        while self.maze[state[0]][state[1]] != 'G':
            action = np.argmax(self.q_table[state[0], state[1]])
            next_state = (state[0] + self.actions[action][0], state[1] + self.actions[action][1])
            path.append(next_state)
            state = next_state
            print(state)
        return path

# Example maze
maze = [
    ['#', '#', '#', '#', '#', '#', '#', '#', '#', '#', '#', '#', '#'],
    ['#', 'S', ' ', ' ', '#', ' ', ' ', ' ', ' ', ' ', ' ', ' ', '#'],
    ['#', '#', ' ', '#', '#', '#', '#', '#', '#', '#', '#', ' ', '#'],
    ['#', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', '#'],
    ['#', ' ', '#', '#', '#', '#', ' ', '#', '#', '#', '#', ' ', '#'],
    ['#', 'G', ' ', ' ', ' ', '#', ' ', ' ', ' ', ' ', '#', ' ', '#'],
    ['#', '#', '#', '#', ' ', '#', '#', '#', '#', ' ', '#', ' ', '#'],
    ['#', '#', ' ', '#', ' ', ' ', ' ', ' ', '#', '', ' ', ' ', '#'],
    ['#', ' ', ' ', '#', ' ', '#', '#', ' ', '#', '#', '#', '#', '#'],
    ['#', ' ', '#', '#', '#', '#', '#', ' ', '#', ' ', ' ', ' ', '#'],
    ['#', ' ', '#', ' ', ' ', ' ', ' ', ' ', ' ', ' ', '#', '#', '#'],
    ['#', ' ', '#', '#', '#', '#', '#', '#', '#', ' ', ' ', '#', '#'],
    ['#', '#', '#', '#', '#', '#', '#', '#', '#', '#', '#', '#', '#']
]

# Create Q-learning agent
agent = QLearningMazeSolver(maze)

# Train the agent
start = (1, 1)  # Start position
agent.train(start)

# Find path
path = agent.find_path(start)
print("Path found by Q-learning agent:", path)
