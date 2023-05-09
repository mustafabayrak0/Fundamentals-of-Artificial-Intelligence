import numpy as np
from agent.agent import *


### MUSTAFA BAYRAK 150210339

def expand(Agent, node):
    Agent.expanded_node += 1
    Agent.empty_tile = node.position
    children = []

    for direction in Agent.directions:
        new_position = (Agent.empty_tile[0] + direction[0], Agent.empty_tile[1] + direction[1])

        if 0 <= new_position[0] < Agent.game_size and 0 <= new_position[1] < Agent.game_size:
            new_matrix = [list(r) for r in node.matrix]
            new_matrix[Agent.empty_tile[0]][Agent.empty_tile[1]], new_matrix[new_position[0]][new_position[1]] = \
                new_matrix[new_position[0]][new_position[1]], new_matrix[Agent.empty_tile[0]][Agent.empty_tile[1]]

            if new_matrix not in Agent.frontier and new_matrix not in Agent.explored:
                Agent.generated_node += 1
                g_score = node.g_score + 1
                h_score = manhattan_distance(Agent, new_matrix)
                child_node = Node(node, new_position, new_matrix, g_score, h_score)
                children.append(child_node)

    return children


# Priority Queue is not iterable, therefore I defined new expand function for A*
def expandAStar(Agent, node):
    Agent.expanded_node += 1
    Agent.empty_tile = node.position
    children = []

    for direction in Agent.directions:
        new_position = (Agent.empty_tile[0] + direction[0], Agent.empty_tile[1] + direction[1])

        if 0 <= new_position[0] < Agent.game_size and 0 <= new_position[1] < Agent.game_size:
            new_matrix = [list(r) for r in node.matrix]
            new_matrix[Agent.empty_tile[0]][Agent.empty_tile[1]], new_matrix[new_position[0]][new_position[1]] = \
                new_matrix[new_position[0]][new_position[1]], new_matrix[Agent.empty_tile[0]][Agent.empty_tile[1]]

            if new_matrix not in Agent.explored and not Agent.frontier.contains(new_matrix):
                Agent.generated_node += 1
                g_score = node.g_score + 1
                h_score = diagonal_distance(Agent, new_matrix)
                child_node = Node(node, new_position, new_matrix, g_score, h_score)
                children.append(child_node)

    return children


def manhattan_distance(Agent, matrix):
    distance = 0
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            if matrix[i][j] != 0:
                goal_position = Agent.find_tile_position(Agent.desired_matrix, matrix[i][j])
                distance += abs(i - goal_position[0]) + abs(j - goal_position[1])
    return distance


def euclidean_distance(Agent, matrix):
    distance = 0
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            if matrix[i][j] != 0:
                goal_position = Agent.find_tile_position(Agent.desired_matrix, matrix[i][j])
                distance += np.sqrt((i - goal_position[0]) ** 2 + (j - goal_position[1]) ** 2)
    return distance


def diagonal_distance(Agent, matrix):
    distance = 0
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            if matrix[i][j] != 0:
                goal_position = Agent.find_tile_position(Agent.desired_matrix, matrix[i][j])
                d_x, d_y = abs(i - goal_position[0]), abs(j - goal_position[1])
                distance += max(d_x, d_y)
    return distance


class BFSAgent(Agent):
    def __init__(self, matrix):
        """
            Initializes the BFS agent class.

            Args:
                matrix (array): Initial game matrix
        """
        # Initializing the parent class
        super().__init__(matrix)
        # Initializing the frontier as a FIFO queue
        self.frontier = []
        self.explored = []

    def tree_solve(self):
        """
            Solves the game using tree base BFS algorithm.

            Returns:
                list: A list of game matrices that represent the solution.
        """
        ### YOUR CODE HERE ###
        node = Node(None, self.empty_tile, self.initial_matrix)
        if self.checkEqual(node.matrix, self.desired_matrix):
            return self.get_moves(node)
        self.frontier.append(node)
        while self.frontier:
            n = self.frontier.pop(0)
            for child in expand(self, n):
                if self.checkEqual(child.matrix, self.desired_matrix):
                    self.maximum_node_in_memory = self.expanded_node + self.generated_node
                    return self.get_moves(child)
                self.frontier.append(child)
        self.maximum_node_in_memory = self.expanded_node + self.generated_node
        return None

        ### YOUR CODE HERE ###

    def graph_solve(self):
        """
            Solves the game using graph base BFS algorithm.

            Returns:
                list: A list of game matrices that represent the solution.
        """
        ### YOUR CODE HERE ###

        node = Node(None, self.empty_tile, self.initial_matrix)
        if self.checkEqual(node.matrix, self.desired_matrix):
            return self.get_moves(node)
        self.frontier.append(node)
        self.explored.append(node.matrix)
        while self.frontier:
            n = self.frontier.pop(0)
            for child in expand(self, n):
                if self.checkEqual(child.matrix, self.desired_matrix):
                    self.maximum_node_in_memory = len(self.frontier) + len(
                        self.explored) + self.expanded_node + self.generated_node
                    return self.get_moves(child)
                if child.matrix not in self.explored:
                    self.frontier.append(child)
                    self.explored.append(child.matrix)
        self.maximum_node_in_memory = len(self.frontier) + len(
            self.explored) + self.expanded_node + self.generated_node
        return None

    ### YOUR CODE HERE ###


class DFSAgent(Agent):
    def __init__(self, matrix):
        """
            Initializes the DFS agent class.

            Args:
                matrix (array): Initial game matrix
        """
        # Initializing the parent class
        super().__init__(matrix)
        self.frontier = []
        self.explored = []

    def tree_solve(self):
        """
            Solves the game using tree base DFS algorithm.

            Returns:
                list: A list of game matrices that represent the solution.
        """
        ### YOUR CODE HERE ###
        node = Node(None, self.empty_tile, self.initial_matrix)
        if self.checkEqual(node.matrix, self.desired_matrix):
            return self.get_moves(node)
        self.frontier.append(node)
        while self.frontier:
            n = self.frontier.pop()
            for child in expand(self, n):
                if self.checkEqual(child.matrix, self.desired_matrix):
                    self.maximum_node_in_memory = self.expanded_node + self.generated_node
                    return self.get_moves(child)
                self.frontier.append(child)
        self.maximum_node_in_memory = self.expanded_node + self.generated_node
        return None
        ### YOUR CODE HERE ###

    def graph_solve(self):
        """
            Solves the game using graph base DFS algorithm.

            Returns:
                list: A list of game matrices that represent the solution.
        """
        ### YOUR CODE HERE ###
        node = Node(None, self.empty_tile, self.initial_matrix)
        if self.checkEqual(node.matrix, self.desired_matrix):
            return self.get_moves(node)
        self.frontier.append(node)
        self.explored.append(node.matrix)
        while self.frontier:
            n = self.frontier.pop()
            for child in expand(self, n):
                if self.checkEqual(child.matrix, self.desired_matrix):
                    self.maximum_node_in_memory = len(self.frontier) + len(
                        self.explored) + self.expanded_node + self.generated_node
                    return self.get_moves(child)
                if child.matrix not in self.explored:
                    self.frontier.append(child)
                    self.explored.append(child.matrix)
        self.maximum_node_in_memory = len(self.frontier) + len(
            self.explored) + self.expanded_node + self.generated_node
        return None
        ### YOUR CODE HERE ###


class AStarAgent(Agent):
    def __init__(self, matrix):
        """
            Initializes the A* agent class.

            Args:
                matrix (array): Initial game matrix
        """
        # Initializing the parent class
        super().__init__(matrix)
        # Initializing the frontier as a priority queue
        self.frontier = PriorityQueue()
        self.explored = []

    def tree_solve(self):
        """
            Solves the game using tree base A* algorithm.

            Returns:
                list: A list of game matrices that represent the solution.
        """
        ### YOUR CODE HERE ###
        node = Node(None, self.empty_tile, self.initial_matrix)
        if self.checkEqual(node.matrix, self.desired_matrix):
            return self.get_moves(node)
        self.frontier.push(node, diagonal_distance(self, node.matrix))
        while not self.frontier.isEmpty():
            n = self.frontier.pop()
            for child in expandAStar(self, n):
                if self.checkEqual(child.matrix, self.desired_matrix):
                    self.maximum_node_in_memory = self.frontier.size() + len(
                        self.explored) + self.expanded_node + self.generated_node
                    return self.get_moves(child)
                self.frontier.push(child, child.f_score)
        self.maximum_node_in_memory = self.frontier.size() + len(
            self.explored) + self.expanded_node + self.generated_node
        return None
        ### YOUR CODE HERE ###

    def graph_solve(self):
        """
            Solves the game using graph base A* algorithm.

            Returns:
                list: A list of game matrices that represent the solution.
        """
        ### YOUR CODE HERE ###
        node = Node(None, self.empty_tile, self.initial_matrix)
        if self.checkEqual(node.matrix, self.desired_matrix):
            return self.get_moves(node)
        self.frontier.push(node, diagonal_distance(self, node.matrix))
        self.explored.append(node.matrix)
        while not self.frontier.isEmpty():
            n = self.frontier.pop()
            for child in expandAStar(self, n):
                if self.checkEqual(child.matrix, self.desired_matrix):
                    self.maximum_node_in_memory = self.frontier.size() + len(
                        self.explored) + self.expanded_node + self.generated_node
                    return self.get_moves(child)
                if child.matrix not in self.explored:
                    self.explored.append(child.matrix)
                    self.frontier.push(child, child.f_score)
        self.maximum_node_in_memory = self.frontier.size() + len(
            self.explored) + self.expanded_node + self.generated_node
        return None
        ### YOUR CODE HERE ###
