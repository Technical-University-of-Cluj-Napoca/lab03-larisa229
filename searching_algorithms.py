import math
from heapq import heappush, heappop

from pygame.transform import threshold

from utils import *
from collections import deque
from queue import PriorityQueue
from grid import Grid
from spot import Spot

def bfs(draw: callable, grid: Grid, start: Spot, end: Spot) -> bool:
    if start is None or end is None:
        return False

    queue = deque([start])
    visited = {start}
    came_from = {}

    while queue:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()

        current = queue.popleft()

        if current == end:
            while current in came_from:
                current = came_from[current]
                current.make_path()
                draw()
            end.make_end()
            start.make_start()
            return True

        for neighbor in current.neighbors:
            if neighbor not in visited and not neighbor.is_barrier():
                visited.add(neighbor)
                came_from[neighbor] = current
                queue.append(neighbor)
                neighbor.make_open()

        draw()

        if current != start:
            current.make_closed()

    return False

def dfs(draw: callable, grid: Grid, start: Spot, end: Spot) -> bool:
    if start is None or end is None:
        return False

    stack = [start]
    visited = {start}
    came_from = {}

    while stack:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return False

        current = stack.pop()

        if current == end:
            while current in came_from:
                current = came_from[current]
                current.make_path()
                draw()
            end.make_end()
            start.make_start()
            return True

        for neighbor in current.neighbors:
            if neighbor not in visited and not neighbor.is_barrier():
                visited.add(neighbor)
                came_from[neighbor] = current
                stack.append(neighbor)
                neighbor.make_open()

        draw()

        if current != start:
            current.make_closed()

    return False

def h_manhattan_distance(p1: tuple[int, int], p2: tuple[int, int]) -> float:
    x1, y1 = p1
    x2, y2 = p2
    return abs(x1 - x2) + abs(y1 - y2)

def h_euclidian_distance(p1: tuple[int, int], p2: tuple[int, int]) -> float:
    x1, y1 = p1
    x2, y2 = p2
    return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)


def astar(draw: callable, grid: Grid, start: Spot, end: Spot) -> bool:

    count = 0
    open_heap = []
    heappush(open_heap, (0, count, start))

    came_from = {}
    g_score = {spot: float('inf') for row in grid.grid for spot in row}
    g_score[start] = 0

    f_score = {spot: float('inf') for row in grid.grid for spot in row}
    f_score[start] = h_manhattan_distance((start.row, start.col), (end.row, end.col))

    lookup_set = {start}

    while open_heap:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()

        current = heappop(open_heap)[2]
        lookup_set.remove(current)

        if current == end:
            while current in came_from:
                current = came_from[current]
                current.make_path()
                draw()
            end.make_end()
            start.make_start()
            return True

        for neighbor in current.neighbors:
            if neighbor.is_barrier():
                continue
            tentative_g = g_score[current] + 1

            if tentative_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score[neighbor] = tentative_g + h_manhattan_distance((neighbor.row, neighbor.col), (end.row, end.col))
                if neighbor not in lookup_set:
                    count += 1
                    heappush(open_heap, (f_score[neighbor], count, neighbor))
                    lookup_set.add(neighbor)
                    neighbor.make_open()

        draw()

        if current != start:
            current.make_closed()

    return False

def dls(draw: callable, grid: Grid, start: Spot, end: Spot, limit: int) -> bool:
    if start is None or end is None:
        return False

    stack = [(start, 0)]
    visited = {start}
    came_from = {}

    while stack:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return False

        current, depth = stack.pop()

        if current == end:
            while current in came_from:
                current = came_from[current]
                current.make_path()
                draw()
            end.make_end()
            start.make_start()
            return True

        if depth < limit:
            for neighbor in current.neighbors:
                if neighbor not in visited and not neighbor.is_barrier():
                    visited.add(neighbor)
                    came_from[neighbor] = current
                    stack.append((neighbor, depth + 1))
                    neighbor.make_open()

            draw()

        if current != start:
            current.make_closed()

    return False

def ucs(draw: callable, grid: Grid, start: Spot, end: Spot) -> bool:
    if start is None or end is None:
        return False

    count = 0
    queue = PriorityQueue()
    queue.put((0, count, start))
    visited = {start}
    came_from = {}
    cost_so_far = {spot: float('inf') for row in grid.grid for spot in row}
    cost_so_far[start] = 0

    while not queue.empty():
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()

        current_cost, _, current = queue.get()
        visited.remove(current)

        if current == end:
            while current in came_from:
                current = came_from[current]
                current.make_path()
                draw()
            end.make_end()
            start.make_start()
            return True

        for neighbor in current.neighbors:
            if neighbor.is_barrier():
                continue

            new_cost = cost_so_far[current] + 1

            if new_cost < cost_so_far[neighbor]:
                cost_so_far[neighbor] = new_cost
                came_from[neighbor] = current
                if neighbor not in visited:
                    count += 1
                    queue.put((new_cost, count, neighbor))
                    visited.add(neighbor)
                    neighbor.make_open()
        draw()

        if current != start:
            current.make_closed()

    return False

def greedy(draw: callable, grid: Grid, start: Spot, end: Spot) -> bool:

    if start is None or end is None:
        return False

    count = 0
    open_heap = []
    heappush(open_heap, (0, count, start))

    came_from = {}
    visited = set()
    lookup_set = {start}

    while open_heap:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()

        current = heappop(open_heap)[2]
        lookup_set.remove(current)

        if current in visited:
            continue

        visited.add(current)

        if current == end:
            while current in came_from:
                current = came_from[current]
                current.make_path()
                draw()
            end.make_end()
            start.make_start()
            return True

        for neighbor in current.neighbors:
            if neighbor.is_barrier() or neighbor in visited:
                continue

            h_score = h_manhattan_distance((neighbor.row, neighbor.col), (end.row, end.col))

            if neighbor not in lookup_set:
                came_from[neighbor] = current
                count += 1
                heappush(open_heap, (h_score, count, neighbor))
                lookup_set.add(neighbor)
                neighbor.make_open()

        draw()

        if current != start:
            current.make_closed()

    return False

def ids(draw: callable, grid: Grid, start: Spot, end: Spot, max_depth: int) -> bool:
    if start is None or end is None:
        return False

    for depth in range(max_depth + 1):
        for row in grid.grid:
            for spot in row:
                if not spot.is_barrier() and spot != start and spot != end:
                    spot.reset()

        draw()

        found = dls(draw, grid, start, end, limit=depth)
        if found:
            return True

    return False

def ida_star(draw: callable, grid: Grid, start: Spot, end: Spot) -> bool:
    if start is None or end is None:
        return False

    threshold = h_manhattan_distance((start.row, start.col), (end.row, end.col))

    while threshold < float('inf'):
        for row in grid.grid:
            for spot in row:
                if not spot.is_barrier() and spot != start and spot != end:
                    spot.reset()

        stack = [(start, 0, [start])]
        min_exceeded = float('inf')

        while stack:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()

            current, g_cost, path = stack.pop()

            f_cost = g_cost + h_manhattan_distance((current.row, current.col), (end.row, end.col))

            if f_cost > threshold:
                min_exceeded = min(min_exceeded, f_cost)
                continue

            if current == end:
                for node in path[1:-1]:
                    node.make_path()
                    draw()
                end.make_end()
                start.make_start()
                return True

            for neighbor in current.neighbors:
                if neighbor.is_barrier() or neighbor in path:
                    continue

                neighbor.make_open()
                stack.append((neighbor, g_cost + 1, path + [neighbor]))

            draw()

            if current != start:
                current.make_closed()

        threshold = min_exceeded

    return False

# and the others algorithms...
# ▢ Depth-Limited Search (DLS)
# ▢ Uninformed Cost Search (UCS)
# ▢ Greedy Search
# ▢ Iterative Deepening Search/Iterative Deepening Depth-First Search (IDS/IDDFS)
# ▢ Iterative Deepening A* (IDA)
# Assume that each edge (graph weight) equals