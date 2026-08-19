# agent.py
import random
import heapq
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
    """Goal-based planning agent: simulates the world offline using BFS, DFS, or UCS
    before committing to any physical action."""

    FACINGS = ['Up', 'Down', 'Left', 'Right']

    def __init__(self):
        self.plan = []
        self.active_algo = 'BFS'

    def get_successors(self, pos, walls, grid_size):
        x, y = pos
        width, height = grid_size
        directions = [('Up', (0, 1)), ('Right', (1, 0)),
                       ('Down', (0, -1)), ('Left', (-1, 0))]
        successors = []
        for action, (dx, dy) in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < width and 0 <= ny < height and (nx, ny) not in walls:
                successors.append((action, (nx, ny), 1))
        return successors

    def bfs_search(self, start_pos, goal_pos, walls, grid_size):
        start, goal = tuple(start_pos), tuple(goal_pos)
        walls = set(tuple(w) for w in walls)
        if start == goal:
            return []

        frontier = deque([(start, [])])
        reached = {start}

        while frontier:
            current, path = frontier.popleft()
            for action, next_pos, cost in self.get_successors(current, walls, grid_size):
                if next_pos in reached:
                    continue
                if next_pos == goal:
                    return path + [action]
                reached.add(next_pos)
                frontier.append((next_pos, path + [action]))

        return None

    def dfs_search(self, start_pos, goal_pos, walls, grid_size):
        start, goal = tuple(start_pos), tuple(goal_pos)
        walls = set(tuple(w) for w in walls)
        if start == goal:
            return []

        frontier = [(start, [])]
        reached = {start}

        while frontier:
            current, path = frontier.pop()
            for action, next_pos, cost in self.get_successors(current, walls, grid_size):
                if next_pos in reached:
                    continue
                if next_pos == goal:
                    return path + [action]
                reached.add(next_pos)
                frontier.append((next_pos, path + [action]))

        return None

    def ucs_search(self, start_pos, goal_pos, walls, grid_size):
        start, goal = tuple(start_pos), tuple(goal_pos)
        walls = set(tuple(w) for w in walls)
        if start == goal:
            return []

        counter = 0
        frontier = [(0, counter, start, [])]
        reached = {start: 0}

        while frontier:
            cost_so_far, _, current, path = heapq.heappop(frontier)
            if current == goal:
                return path

            for action, next_pos, step_cost in self.get_successors(current, walls, grid_size):
                new_cost = cost_so_far + step_cost
                if next_pos not in reached or new_cost < reached[next_pos]:
                    reached[next_pos] = new_cost
                    counter += 1
                    heapq.heappush(frontier, (new_cost, counter, next_pos, path + [action]))

        return None

    def closest_food(self, start, food_list):
        return min(food_list, key=lambda f: abs(f[0] - start[0]) + abs(f[1] - start[1]))

    def _turn_diff(self, current_facing, desired_facing):
        order = self.FACINGS
        return (order.index(desired_facing) - order.index(current_facing)) % 4

    def sense_and_act(self, percept: dict) -> str:
        if not self.plan:
            start = percept['agent_pos']
            food_list = percept['all_food']

            if not food_list:
                return 'turn_left'

            goal = self.closest_food(start, food_list)
            walls = percept['walls']
            grid_size = percept['grid_size']

            if self.active_algo == 'BFS':
                result = self.bfs_search(start, goal, walls, grid_size)
            elif self.active_algo == 'DFS':
                result = self.dfs_search(start, goal, walls, grid_size)
            elif self.active_algo == 'UCS':
                result = self.ucs_search(start, goal, walls, grid_size)
            else:
                result = None

            self.plan = result if result else []

        if not self.plan:
            return 'turn_left'

        desired_direction = self.plan[0]
        current_facing = percept['facing']
        diff = self._turn_diff(current_facing, desired_direction)

        if diff == 0:
            self.plan.pop(0)
            return 'move_forward'
        elif diff == 1:
            return 'turn_right'
        else:
            return 'turn_left'
