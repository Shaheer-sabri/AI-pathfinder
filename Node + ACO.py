import tkinter as tk
from threading import Thread
import random

# Node class representing a grid node
class Node:
    def __init__(self, x, y, walkable):
        self.x = x
        self.y = y
        self.walkable = walkable
        self.g_score = float('inf')
        self.h_score = 0
        self.parent = None

# GUI class for the application
class GUI:
    def __init__(self):
        self.window = tk.Tk()
        self.canvas = tk.Canvas(self.window, width=1000, height=1000)
        self.canvas.pack(side=tk.LEFT)
        self.canvas.bind("<Button-1>", self.handle_click)
        self.start_entry = tk.Entry(self.window)
        self.end_entry = tk.Entry(self.window)
        self.button = tk.Button(self.window, text='Run A*', command=self.find_path)
        self.reset_button = tk.Button(self.window, text='Reset', command=self.reset)
        self.aco_button = tk.Button(self.window, text='Run ACO', command=self.run_aco)
        self.yen_button = tk.Button(self.window, text='Run Yens', command=self.run_yen)
        self.start_entry.pack()
        self.end_entry.pack()
        self.button.pack()
        self.reset_button.pack()
        self.aco_button.pack()
        self.yen_button.pack()
        self.grid = []
        
        # Create a grid of nodes and rectangles on the canvas
        for i in range(20):
            row = []
            for j in range(20):
                node = Node(i, j, True)
                row.append(node)
                self.canvas.create_rectangle(i * 50, j * 50, i * 50 + 50, j * 50 + 50, fill='white')
            self.grid.append(row)
        
        self.window.mainloop()

    def reset(self):
        # Reset the grid to its initial state
        for i in range(20):
            for j in range(20):
                self.grid[i][j].walkable = True
                self.canvas.create_rectangle(i * 50, j * 50, i * 50 + 50, j * 50 + 50, fill='white')

    def handle_click(self, event):
        # Handle the mouse click event to toggle the walkability of a node
        x, y = event.x // 50, event.y // 50
        self.grid[x][y].walkable = not self.grid[x][y].walkable
        fill_color = 'black' if not self.grid[x][y].walkable else 'white'
        self.canvas.create_rectangle(x * 50, y * 50, x * 50 + 50, y * 50 + 50, fill=fill_color)

    def find_path(self):
        # Find the path using A* algorithm based on the start and end coordinates entered by the user
        start = tuple(map(int, self.start_entry.get().split(',')))
        end = tuple(map(int, self.end_entry.get().split(',')))
        path = self.astar(start, end)
        
        # Draw the path on the canvas if a valid path is found
        if path:
            for node in path:
                self.canvas.create_rectangle(node.x * 50, node.y * 50, node.x * 50 + 50, node.y * 50 + 50, fill='green')

    def astar(self, start, end):
        # Implementation of the A* algorithm to find the shortest path
        open_set = set([self.grid[start[0]][start[1]]])
        closed_set = set()
        self.grid[start[0]][start[1]].g_score = 0
        self.grid[start[0]][start[1]].h_score = self.manhattan_distance(start, end)
        
        while open_set:
            current = min(open_set, key=lambda node: node.g_score + node.h_score)
            if current == self.grid[end[0]][end[1]]:
                path = []
                while current:
                    path.append(current)
                    current = current.parent
                return path[::-1]
            
            open_set.remove(current)
            closed_set.add(current)
            
            for neighbor in self.get_neighbors(current):
                if neighbor.walkable and neighbor not in closed_set:
                    tentative_g_score = current.g_score + 1
                    if neighbor in open_set:
                        if tentative_g_score < neighbor.g_score:
                            neighbor.g_score = tentative_g_score
                            neighbor.parent = current
                    else:
                        neighbor.g_score = tentative_g_score
                        neighbor.h_score = self.manhattan_distance((neighbor.x, neighbor.y), end)
                        neighbor.parent = current
                        open_set.add(neighbor)

    def get_neighbors(self, node):
        # Get the neighboring nodes of a given node
        x, y = node.x, node.y
        neighbors = []
        if x > 0:
            neighbors.append(self.grid[x-1][y])
        if x < 19:
            neighbors.append(self.grid[x+1][y])
        if y > 0:
            neighbors.append(self.grid[x][y-1])
        if y < 19:
            neighbors.append(self.grid[x][y+1])
        return neighbors

    def manhattan_distance(self, start, end):
        # Calculate the Manhattan distance between two points
        return abs(end[0] - start[0]) + abs(end[1] - start[1])

    def run_aco(self):
        # Run the Ant Colony Optimization (ACO) algorithm in a separate thread
        start = tuple(map(int, self.start_entry.get().split(',')))
        end = tuple(map(int, self.end_entry.get().split(',')))

        # Clear the canvas before running ACO
        self.canvas.delete("aco_path")

        # Run ACO algorithm in a separate thread
        Thread(target=self.aco, args=(start, end)).start()

    def aco(self, start, end):
        # Implementation of the Ant Colony Optimization (ACO) algorithm
        num_ants = 10
        evaporation_rate = 0.5
        alpha = 1
        beta = 2
        Q = 100
        num_iterations = 100

        pheromone = [[1] * 20 for _ in range(20)]

        def calculate_probability(node, next_node):
            # Calculate the probability of an ant moving from the current node to the next node
            pheromone_value = pheromone[next_node.x][next_node.y]
            distance = self.manhattan_distance((node.x, node.y), (next_node.x, next_node.y))
            return pow(pheromone_value, alpha) * pow(1 / distance, beta)

        def update_pheromone(trail):
            # Update the pheromone levels based on the trail of the best ant
            for i in range(20):
                for j in range(20):
                    pheromone[i][j] *= (1 - evaporation_rate)
            for node in trail:
                pheromone[node.x][node.y] += Q

        def select_next_node(current_node, visited):
            # Select the next node for an ant to move to
            neighbors = self.get_neighbors(current_node)
            unvisited_neighbors = [n for n in neighbors if n not in visited and n.walkable]
            if not unvisited_neighbors:
                return None
            probabilities = [calculate_probability(current_node, neighbor) for neighbor in unvisited_neighbors]
            total = sum(probabilities)
            probabilities = [p / total for p in probabilities]
            return random.choices(unvisited_neighbors, probabilities)[0]

        def ant_colony_optimization():
            # Run the Ant Colony Optimization algorithm to find the best path
            best_path = None
            best_distance = float('inf')
            for _ in range(num_iterations):
                ants = [Node(start[0], start[1], True) for _ in range(num_ants)]
                for ant in ants:
                    trail = [ant]
                    visited = set()
                    while ant != self.grid[end[0]][end[1]]:
                        ant = select_next_node(ant, visited)
                        if not ant:
                            break
                        trail.append(ant)
                        visited.add(ant)
                    if trail and trail[-1] == self.grid[end[0]][end[1]]:
                        distance = len(trail) - 1
                        if distance < best_distance:
                            best_distance = distance
                            best_path = trail
            if best_path:
                update_pheromone(best_path)

            # Update the GUI with the path found by ACO
            self.window.after(0, self.update_aco_path, best_path)

        # Run ACO algorithm in a separate thread
        ant_colony_optimization()

    def update_aco_path(self, path):
        # Update the GUI with the path found by ACO
        if path:
            for node in path:
                self.canvas.create_rectangle(node.x * 50, node.y * 50, node.x * 50 + 50, node.y * 50 + 50, fill='orange',
                                             tags="aco_path")

    def run_yen(self):
        # Run Yen's algorithm in a separate thread
        start = tuple(map(int, self.start_entry.get().split(',')))
        end = tuple(map(int, self.end_entry.get().split(',')))

        # Clear the canvas before running Yen's algorithm
        self.canvas.delete("yen_path")

        # Run Yen's algorithm in a separate thread
        Thread(target=self.yen, args=(start, end)).start()

    def yen(self, start, end):
        # Implementation of Yen's algorithm for finding K shortest paths
        def find_shortest_path(start, end):
            # Find the shortest path using A* algorithm
            path = self.astar(start, end)
            if path:
                return path, len(path) - 1
            return None, float('inf')

        def remove_edge(graph, u, v):
            # Remove an edge from the graph
            del graph[u][v]
            del graph[v][u]

        def add_edge(graph, u, v, weight):
            # Add an edge to the graph
            if u in graph:
                graph[u][v] = weight
            else:
                graph[u] = {v: weight}
            if v in graph:
                graph[v][u] = weight
            else:
                graph[v] = {u: weight}

        def find_kth_shortest_path(start, end, k):
            # Find the k-th shortest path from start to end
            graph = {}
            path_list = []

            while k > 0:
                if k == 1:
                    path, _ = find_shortest_path(start, end)
                    if path:
                        path_list.append(path)
                else:
                    for i in range(len(path_list[-1]) - 1):
                        spur_node = path_list[-1][i]
                        root_path = path_list[-1][:i + 1]
                        edges_removed = []

                        for path in path_list:
                            if len(path) > i and root_path == path[:i + 1]:
                                u = path[i]
                                v = path[i + 1]
                                if (u, v) in graph:
                                    weight = graph[u][v]
                                    remove_edge(graph, u, v)
                                    edges_removed.append((u, v, weight))

                        for node in root_path:
                            if node != spur_node:
                                remove_edge(graph, node, spur_node)

                        spur_path, spur_path_cost = find_shortest_path(spur_node, end)

                        if spur_path:
                            total_path = root_path[:-1] + spur_path
                            total_path_cost = len(total_path) - 1
                            path_list.append(total_path)

                        for u, v, weight in edges_removed:
                            add_edge(graph, u, v, weight)

                        for node in root_path:
                            if node != spur_node:
                                add_edge(graph, node, spur_node, self.manhattan_distance(node, spur_node))

                if not path_list:
                    break

                path_list.sort(key=lambda p: len(p) - 1)
                k -= 1

            if path_list:
                shortest_path = path_list[0]
                return shortest_path

            return None

        # Run Yen's algorithm to find the shortest path
        shortest_path = find_kth_shortest_path(start, end, 1)

        # Update the GUI with the path found by Yen's algorithm
        self.window.after(0, self.update_yen_path, shortest_path)

    def update_yen_path(self, path):
        # Update the GUI with the path found by Yen's algorithm
        if path:
            for node in path:
                self.canvas.create_rectangle(node.x * 50, node.y * 50, node.x * 50 + 50, node.y * 50 + 50, fill='blue',
                                             tags="yen_path")

if __name__ == '__main__':
    gui = GUI()
