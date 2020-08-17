import pygame
import math
from queue import PriorityQueue, Queue

WIDTH = 800
HEIGHT = 1000
DIFF = HEIGHT-WIDTH

pygame.display.set_caption("Path Finding Visualizer")
WIN = pygame.display.set_mode((WIDTH, WIDTH))

AQUA = (0,255,255)
CRIMSON = (220,20,60)
NAVY = (0,0,205)
GOLD = (255, 215, 0)
LIGHT_GREEN = (127,255,0)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREY = (128, 128, 128)

class Node:
    def __init__(self, row, col, width, total_rows):
        self.row = row
        self.col = col
        self.x = row * width
        self.y = col * width
        self.color = WHITE
        self.neighbors = []
        self.diff = DIFF
        self.width = width
        self.total_rows = total_rows

    def get_pos(self):
        return self.row, self.col

    def is_closed(self):
        return self.color == NAVY

    def is_open(self):
        return self.color == AQUA

    def is_barrier(self):
        return self.color == BLACK

    def is_start(self):
        return self.color == GOLD

    def is_end(self):
        return self.color == CRIMSON

    def reset(self):
        self.color = WHITE

    def make_start(self):
        self.color = GOLD
    
    def make_closed(self):
        self.color = NAVY

    def make_open(self):
        self.color = AQUA

    def make_barrier(self):
        self.color = BLACK

    def make_end(self):
        self.color = CRIMSON

    def make_path(self):
        self.color = LIGHT_GREEN

    def draw(self, win):
        pygame.draw.rect(win, self.color, (self.x, self.y +
                                           self.diff, self.width, self.width + self.diff))

    def update_neighbors(self, grid):
        self.neighbors = []
        # DOWN
        if self.row < self.total_rows - 1 and not grid[self.row + 1][self.col].is_barrier():
            self.neighbors.append(grid[self.row + 1][self.col])

        # UP
        if self.row > 0 and not grid[self.row - 1][self.col].is_barrier():  
            self.neighbors.append(grid[self.row - 1][self.col])

        # RIGHT
        if self.col < self.total_rows - 1 and not grid[self.row][self.col + 1].is_barrier():
            self.neighbors.append(grid[self.row][self.col + 1])
        
        # LEFT
        if self.col > 0 and not grid[self.row][self.col - 1].is_barrier():  
            self.neighbors.append(grid[self.row][self.col - 1])

    def __lt__(self, other):
        return False


def h(p1, p2):
    # Manhatten distance / taxi cab distance
    x1, y1 = p1
    x2, y2 = p2
    return abs(x1 - x2) + abs(y1 - y2)


def h2(p1, p2):
    # Euclidean distance 
    x1, y1 = p1
    x2, y2 = p2
    c = (abs(x1 - x2))^2 + (abs(y1 - y2))^2
    return math.sqrt(c)
    
def h3(p1, p2):
    # Civ distance
    x1, y1 = p1
    x2, y2 = p2
    return (max(abs(x1 - x2), abs(y1 - y2)))


def reconstruct_path(came_from, current, draw):
    # Returns the path found
    while current in came_from:
        current = came_from[current]
        current.make_path()
        draw()

def make_grid(rows, width):
    grid = []
    gap = width // rows
    for i in range(rows):
        grid.append([])
        for j in range(rows):
            node = Node(i, j, gap, rows)
            grid[i].append(node)

    return grid

def draw_grid(win, rows, width, diff):
    gap = width // rows
    for i in range(rows):
        pygame.draw.line(win, GREY, (0, i * gap + diff),
                         (width, i * gap + diff))
        for j in range(rows):
            pygame.draw.line(win, GREY, (j * gap, diff),
                             (j * gap, width + diff))


def set_text_box(win):
    pygame.init()
    image = pygame.image.load('vis_keys.png')
    image = pygame.transform.scale(image, (800, 200))
    win.blit(image,(0,0))


def draw(win, grid, rows, width, diff):
    win.fill(WHITE)

    for row in grid:
        for node in row:
            node.draw(win)

    set_text_box(win)
    draw_grid(win, rows, width, diff)
    pygame.display.update()


def get_clicked_pos(pos, rows, width, diff):

    x, y = pos
    if y > diff:
        gap = width // rows
        row = x // gap
        col = (y-diff) // gap
        return row, col

    return False


###########################

### ALGORITHMS #####

###########################

def a_star_m(draw, grid, start, end):
    count = 0
    queue = PriorityQueue()
    queue.put((0, count, start))
    came_from = {}
    g_score = {node: float("inf") for row in grid for node in row}
    g_score[start] = 0
    f_score = {node: float("inf") for row in grid for node in row}
    f_score[start] = h(start.get_pos(), end.get_pos())

    queue_hash = {start}

    while not queue.empty():
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()

        current = queue.get()[2]
        queue_hash.remove(current)

        if current == end:
            reconstruct_path(came_from, end, draw)
            end.make_end()
            return True

        for neighbor in current.neighbors:
            temp_g_score = g_score[current] + 1

            if temp_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = temp_g_score
                f_score[neighbor] = temp_g_score + h(neighbor.get_pos(), end.get_pos())
                if neighbor not in queue_hash:
                    count += 1
                    queue.put((f_score[neighbor], count, neighbor))
                    queue_hash.add(neighbor)
                    neighbor.make_open()

        draw()

        if current != start:
            current.make_closed()

    return False


def a_star_e(draw, grid, start, end):
    count = 0
    queue = PriorityQueue()
    queue.put((0, count, start))
    came_from = {}
    g_score = {node: float("inf") for row in grid for node in row}
    g_score[start] = 0
    f_score = {node: float("inf") for row in grid for node in row}
    f_score[start] = h(start.get_pos(), end.get_pos())

    queue_hash = {start}

    while not queue.empty():
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()

        current = queue.get()[2]
        queue_hash.remove(current)

        if current == end:
            reconstruct_path(came_from, end, draw)
            end.make_end()
            return True

        for neighbor in current.neighbors:
            temp_g_score = g_score[current] + 1

            if temp_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = temp_g_score
                f_score[neighbor] = temp_g_score + h2(neighbor.get_pos(), end.get_pos())
                if neighbor not in queue_hash:
                    count += 1
                    queue.put((f_score[neighbor], count, neighbor))
                    queue_hash.add(neighbor)
                    neighbor.make_open()

        draw()

        if current != start:
            current.make_closed()

    return False

def a_star_c(draw, grid, start, end):
    count = 0
    queue = PriorityQueue()
    queue.put((0, count, start))
    came_from = {}
    g_score = {node: float("inf") for row in grid for node in row}
    g_score[start] = 0
    f_score = {node: float("inf") for row in grid for node in row}
    f_score[start] = h(start.get_pos(), end.get_pos())

    queue_hash = {start}

    while not queue.empty():
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()

        current = queue.get()[2]
        queue_hash.remove(current)

        if current == end:
            reconstruct_path(came_from, end, draw)
            end.make_end()
            return True

        for neighbor in current.neighbors:
            temp_g_score = g_score[current] + 1

            if temp_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = temp_g_score
                f_score[neighbor] = temp_g_score + h3(neighbor.get_pos(), end.get_pos())
                if neighbor not in queue_hash:
                    count += 1
                    queue.put((f_score[neighbor], count, neighbor))
                    queue_hash.add(neighbor)
                    neighbor.make_open()

        draw()

        if current != start:
            current.make_closed()

    return False

def dfs(draw, grid, start, end):
    
    def recursive_dfs(node, visited, came_from):
       
        visited.append(node)
        node.make_open()
        draw()

        if node == end:
            reconstruct_path(came_from, end, draw)
            end.make_end()
            return True 
        
        for neighbor in node.neighbors:
            if neighbor not in visited:
                came_from[neighbor] = node
                p = recursive_dfs(neighbor, visited, came_from)
                if p:
                    return p
    
        return False

    came_from = {}
    visited = []
    return recursive_dfs(start, visited, came_from)

def bfs(draw, start, end):

    queue = Queue()
    visited = []
    came_from = {}
    queue.put(start)
    visited.append(start)

    while not queue.empty():
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()

        current = queue.get()

        if current == end:
            reconstruct_path(came_from, end, draw)
            end.make_end()
            return True

        for neighbor in current.neighbors:
            
            if neighbor not in visited:
                visited.append(neighbor)
                queue.put(neighbor)
                came_from[neighbor] = current
                neighbor.make_open()

        draw()

        if current != start:
            current.make_closed()

    return False

def dijkstra(draw, grid, start, end):
    count = 0
    queue = PriorityQueue()
    queue.put((0, count, start))
    came_from = {}
    g_score = {node: float("inf") for row in grid for node in row}
    g_score[start] = 0

    queue_hash = {start}

    while not queue.empty():
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()

        current = queue.get()[2]
        queue_hash.remove(current)

        if current == end:
            reconstruct_path(came_from, end, draw)
            end.make_end()
            return True

        for neighbor in current.neighbors:
            temp_g_score = g_score[current] + 1

            if temp_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = temp_g_score
                if neighbor not in queue_hash:
                    count += 1
                    queue.put((g_score[neighbor], count, neighbor))
                    queue_hash.add(neighbor)
                    neighbor.make_open()

        draw()

        if current != start:
            current.make_closed()

    return False

def swarm(draw, grid, start, end):
    count = 0
    count2 = 0
    set_1 = PriorityQueue()
    set_2 = PriorityQueue()
    set_1.put((0, count, start))
    set_2.put((0, count2, end))
    came_from = {}
    came_from2 = {}
    g_score = {node: float("inf") for row in grid for node in row}
    g_score[start] = 0
    f_score = {node: float("inf") for row in grid for node in row}
    f_score[start] = h(start.get_pos(), end.get_pos())
    g_score2 = {node: float("inf") for row in grid for node in row}
    g_score2[end] = 0
    f_score2 = {node: float("inf") for row in grid for node in row}
    f_score2[end] = h(end.get_pos(), start.get_pos())
    set_1_hash = {start}
    set_2_hash = {end}
    

    while not set_1.empty() or not set_2.empty():
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()

        curr_1 = set_1.get()[2]
        curr_2 = set_2.get()[2]
        set_1_hash.remove(curr_1)
        set_2_hash.remove(curr_2)

        for i in set_1_hash:
            for j in set_2_hash:
                if i == j:
                    i.make_path()
                    reconstruct_path(came_from, i, draw)
                    reconstruct_path(came_from2, j, draw)
                    end.make_end()
                    return True

        if curr_1 == end:
            reconstruct_path(came_from, end, draw)
            end.make_end()
            return True


        for neighbor in curr_1.neighbors:
            temp_g_score = g_score[curr_1] + 1
            if temp_g_score < g_score[neighbor]:
                came_from[neighbor] = curr_1
                g_score[neighbor] = temp_g_score
                f_score[neighbor] = temp_g_score + h(neighbor.get_pos(), end.get_pos())
                if neighbor not in set_1_hash:
                    count += 1
                    set_1.put((f_score[neighbor], count, neighbor))
                    set_1_hash.add(neighbor)
                    neighbor.make_open()
        
        for neighbor in curr_2.neighbors:
                temp_g_score = g_score2[curr_2] + 1

                if temp_g_score < g_score2[neighbor]:
                    came_from2[neighbor] = curr_2
                    g_score2[neighbor] = temp_g_score
                    f_score2[neighbor] = temp_g_score + h(neighbor.get_pos(), start.get_pos())
                    if neighbor not in set_2_hash:
                        count2 += 1
                        set_2.put((f_score2[neighbor], count2, neighbor))
                        set_2_hash.add(neighbor)
                        neighbor.make_open()

        draw()
        if curr_1 != start:
            curr_1.make_closed()
        if curr_2 != end:
            curr_2.make_closed()
    return False

###########################

### MAIN #####

###########################

def main(win, width, diff):
    ROWS = 40
    grid = make_grid(ROWS, width)

    start = None
    end = None

    run = True
    while run:
        draw(win, grid, ROWS, width, diff)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False

            if pygame.mouse.get_pressed()[0]:  # LEFT
                pos = pygame.mouse.get_pos()
                location = get_clicked_pos(pos, ROWS, width, diff)
                if not type(location) == bool:
                    r = location[0]
                    c = location[1]
                    node = grid[r][c]
                    if not start and node != end:
                        start = node
                        start.make_start()

                    elif not end and node != start:
                        end = node
                        end.make_end()

                    elif node != end and node != start:
                        node.make_barrier()

            elif pygame.mouse.get_pressed()[2]:  # RIGHT
                pos = pygame.mouse.get_pos()
                location = get_clicked_pos(pos, ROWS, width, diff)
                if len(location) == 2:  # (not type(location) == bool)
                    r = location[0]
                    c = location[1]
                    node = grid[r][c]
                    node.reset()
                    if node == start:
                        start = None
                    elif node == end:
                        end = None

            if event.type == pygame.KEYDOWN:
                
                if event.key == pygame.K_m and start and end:
                    for row in grid:
                        for node in row:
                            node.update_neighbors(grid)

                    a_star_m(lambda: draw(win, grid, ROWS,
                                        width, diff), grid, start, end)


                if event.key == pygame.K_e and start and end:
                    for row in grid:
                        for node in row:
                            node.update_neighbors(grid)

                    a_star_e(lambda: draw(win, grid, ROWS,
                                               width, diff), grid, start, end)

                if event.key == pygame.K_a and start and end:
                    for row in grid:
                        for node in row:
                            node.update_neighbors(grid)

                    a_star_c(lambda: draw(win, grid, ROWS,
                                               width, diff), grid, start, end)

                if event.key == pygame.K_d and start and end:
                    for row in grid:
                        for node in row:
                            node.update_neighbors(grid)

                    dijkstra(lambda: draw(win, grid, ROWS,
                                          width, diff), grid, start, end)

                if event.key == pygame.K_b and start and end:
                    for row in grid:
                        for node in row:
                            node.update_neighbors(grid)

                    bfs(lambda: draw(win, grid, ROWS, width, diff), start, end)

                if event.key == pygame.K_f and start and end:
                    for row in grid:
                        for node in row:
                            node.update_neighbors(grid)

                    dfs(lambda: draw(win, grid, ROWS, width, diff), grid, start, end)

                if event.key == pygame.K_s and start and end:
                    for row in grid:
                        for node in row:
                            node.update_neighbors(grid)

                    swarm(lambda: draw(win, grid, ROWS, width, diff), grid, start, end)


                if event.key == pygame.K_c:
                    start = None
                    end = None
                    grid = make_grid(ROWS, width)

    pygame.quit()

main(WIN, WIDTH, DIFF)