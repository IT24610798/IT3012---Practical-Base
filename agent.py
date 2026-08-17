# agent.py
import random
from collections import deque


class GreedyGridAgent:
    """A simple agent that tries to move around systematically to clear the grid."""

    def __init__(self):
        self.actions_pool = ['Up', 'Down', 'Left', 'Right']

    def sense_and_act(self, percept: dict) -> str:
        pos = percept.get('agent_pos', [0, 0])
        remaining_food = percept.get('remaining_food', 0)

        if remaining_food == 0:
            return 'Up'

        if percept.get('hit_wall'):
            return random.choice(['Left', 'Right', 'Down', 'Up'])

        food_pos = percept.get('food_pos')
        if food_pos is not None:
            fx, fy = food_pos
            x, y = pos
            if fx > x:
                return 'Right'
            if fx < x:
                return 'Left'
            if fy > y:
                return 'Up'
            if fy < y:
                return 'Down'

        return random.choice(self.actions_pool)


class SimpleReflexAgent:
    """Very small reactive agent: avoid walls, otherwise keep moving."""

    def __init__(self):
        self.last_action = None

    def sense_and_act(self, percept: dict) -> str:
        if percept.get('wall_ahead', False):
            action = 'Left' if self.last_action != 'Left' else 'Right'
            self.last_action = action
            return action

        action = 'Up' if self.last_action != 'Up' else 'Right'
        self.last_action = action
        return action


class ModelBasedAgent:
    """A minimal stateful agent that changes action when the same percept repeats."""

    def __init__(self):
        self.last_action = None
        self.last_percept = None

    def sense_and_act(self, percept: dict) -> str:
        if self.last_percept == percept and percept.get('wall_ahead', False):
            choices = ['Up', 'Down', 'Left', 'Right']
            if self.last_action in choices:
                choices.remove(self.last_action)
            action = choices[0]
        elif percept.get('wall_ahead', False):
            action = 'Left' if self.last_action != 'Left' else 'Right'
        else:
            action = 'Up'

        self.last_action = action
        self.last_percept = percept
        return action


class SearchAgent:
    """Problem-solving agent using breadth-first search to find the shortest path."""

    def bfs_search(self, start_pos, goal_pos, walls, grid_size):
        start = tuple(start_pos)
        goal = tuple(goal_pos)
        width, height = grid_size

        if start == goal:
            return []

        walls = set(tuple(pos) for pos in walls)
        queue = deque([(start, [])])
        visited = {start}

        directions = [
            ('Up', (0, 1)),
            ('Right', (1, 0)),
            ('Down', (0, -1)),
            ('Left', (-1, 0))
        ]

        while queue:
            current, path = queue.popleft()
            x, y = current

            for action, (dx, dy) in directions:
                nx, ny = x + dx, y + dy
                next_pos = (nx, ny)

                if not (0 <= nx < width and 0 <= ny < height):
                    continue
                if next_pos in walls or next_pos in visited:
                    continue

                if next_pos == goal:
                    return path + [action]

                visited.add(next_pos)
                queue.append((next_pos, path + [action]))

        return None