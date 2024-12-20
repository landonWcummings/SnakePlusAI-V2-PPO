import gymnasium as gym
from gymnasium import spaces
import numpy as np
from collections import deque
import random
import pygame
import time

class SnakeEnv(gym.Env):
    metadata = {'render_modes': ['human']}

    def __init__(self, gridsize=6, headless=False):
        super(SnakeEnv, self).__init__()
        self.gridsize = gridsize
        self.headless = headless
        self.inittime = time.time()
        self.changereward = False
        
        # The observation now includes the NxN grid plus 4 boolean values, 
        # so its length is gridsize*gridsize + 4.
        # Grid values: 0 to 100
        # Booleans: 0 or 1
        self.observation_space = spaces.Box(
            low=0,
            high=100,
            shape=((self.gridsize*self.gridsize)+4,),
            dtype=np.uint8
        )
        self.action_space = spaces.Discrete(4)  # 0: Up, 1: Right, 2: Down, 3: Left

        if not self.headless:
            pygame.init()
            self.window_width = 600
            self.window_height = 700
            self.window = pygame.display.set_mode((self.window_width, self.window_height))
        # Initialize state
        self.reset()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.gamegrid = np.zeros((self.gridsize, self.gridsize), dtype=np.uint8)
        self.score = 0
        self.done = False
        self.steps = 0
        self.static_states = 0
        self.snake = deque()

        # Place the snake in a random position with length 2
        empty_cells = [(x, y) for x in range(self.gridsize) for y in range(self.gridsize)]
        head = random.choice(empty_cells)
        empty_cells.remove(head)

        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        random.shuffle(directions)

        placed = False
        for dx, dy in directions:
            tail = (head[0] + dx, head[1] + dy)
            if 0 <= tail[0] < self.gridsize and 0 <= tail[1] < self.gridsize and tail not in self.snake:
                self.snake.appendleft(head)
                self.snake.append(tail)
                placed = True
                break

        if not placed:
            # If we fail to place second segment, just reset again
            return self.reset()

        # Set initial direction
        self.direction = (self.snake[0][0] - self.snake[1][0], self.snake[0][1] - self.snake[1][1])
        self.place_food()
        self.update_grid()
        observation = self.get_observation()
        info = {}
        return observation, info

    def linear_alpha_schedule(self):
        # Linear decay of alpha from 1 to 0 over total_timesteps
        transition_time = 3000
        curt = time.time()
        dif = curt - self.inittime
        return max(0, 1 - dif / transition_time)

    def step(self, action):
        if self.done:
            # If already done, just return same state
            observation = self.get_observation()
            return observation, 0.0, True, False, {}

        self.steps += 1
        dir_map = {0: (0, -1), 1: (1, 0), 2: (0, 1), 3: (-1, 0)}
        new_direction = dir_map[int(action)]

        # Prevent going opposite direction
        if (new_direction[0]*-1, new_direction[1]*-1) == self.direction and len(self.snake) > 1:
            new_direction = self.direction
        else:
            self.direction = new_direction

        # Calculate new head
        head_x, head_y = self.snake[0]
        dx, dy = self.direction
        new_head = (head_x + dx, head_y + dy)

        if self.changereward:
            alpha = self.linear_alpha_schedule()
            oldr = -0.01

        reward = -0.005  # small negative step cost

        if self.is_collision(new_head):
            self.done = True
            reward = -0.9
            if self.changereward:
                oldr = -1
        else:
            self.snake.appendleft(new_head)
            # Check if food eaten
            if new_head == self.food_pos:
                reward = 1
                if self.changereward:
                    oldr = 1
                self.score += 1
                self.place_food()
                self.static_states = 0
            else:
                self.snake.pop()
                self.static_states += 1

            self.update_grid()

            if self.static_states > 220:
                reward = -1
                self.done = True

        if self.changereward:
            combined_reward = alpha * oldr + (1 - alpha) * reward
        else:
            combined_reward = reward

        observation = self.get_observation()

        terminated = self.done
        truncated = False
        info = {}
        return observation, combined_reward, terminated, truncated, info

    def render(self):
        if self.headless:
            return
        cell_size = self.window_width // self.gridsize
        self.window.fill((0, 0, 0))
        for x in range(self.gridsize):
            for y in range(self.gridsize):
                rect = pygame.Rect(x * cell_size, y * cell_size, cell_size, cell_size)
                if self.gamegrid[y][x] == 1:
                    pygame.draw.rect(self.window, (255, 0, 0), rect)  # Food
                elif self.gamegrid[y][x] == 2:
                    pygame.draw.rect(self.window, (0, 255, 0), rect)  # Snake head
                elif self.gamegrid[y][x] > 2:
                    pygame.draw.rect(self.window, (20, 200, 20), rect)  # Snake body
        pygame.display.flip()
        pygame.event.pump()

    def close(self):
        if not self.headless:
            pygame.quit()

    def is_collision(self, position):
        x, y = position
        if x < 0 or x >= self.gridsize or y < 0 or y >= self.gridsize:
            return True
        if position in list(self.snake):
            return True
        return False

    def place_food(self):
        empty_cells = [(i, j) for i in range(self.gridsize) for j in range(self.gridsize) if (i, j) not in self.snake]

        if not empty_cells:
            # No space to place food, end episode
            self.done = True
            return

        self.food_pos = random.choice(empty_cells)

    def update_grid(self):
        self.gamegrid.fill(0)
        # Snake
        for idx, (x, y) in enumerate(self.snake):
            if idx == len(self.snake) - 1:  # Tail
                self.gamegrid[y][x] = 100
            else:
                self.gamegrid[y][x] = 2 + idx

        # Food
        fx, fy = self.food_pos
        self.gamegrid[fy][fx] = 1

    def get_observation(self):
        # Flatten the grid
        flat_grid = self.gamegrid.flatten()

        # Check the four directions: Up, Right, Down, Left relative to the snake's head
        head_x, head_y = self.snake[0]
        directions = {
            'up':    (head_x, head_y - 1),
            'right': (head_x + 1, head_y),
            'down':  (head_x, head_y + 1),
            'left':  (head_x - 1, head_y)
        }

        danger_values = []
        for d in ['up', 'right', 'down', 'left']:
            if self.is_collision(directions[d]):
                danger_values.append(1)
            else:
                danger_values.append(0)

        # Append these four values to the flattened grid observation
        observation = np.concatenate([flat_grid, np.array(danger_values, dtype=np.uint8)], axis=0)
        return observation
