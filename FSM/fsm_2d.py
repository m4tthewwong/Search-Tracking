import numpy as np
from PIL import Image
import itertools
from threading import Thread
    
def fsm_2d(grid, f, h):
    (ny, nx) = f.shape
    
    def new_grid(i, j, y_min, x_min):
        cost = 0
        c = f[i, j] * h
        diff = y_min - x_min
        if abs(diff) > c:
            cost = np.minimum(y_min, x_min) + c
        else:
            cost = 0.5 * (y_min + x_min + np.sqrt(2.0 * (f[i, j]**2) * (h**2) - (diff)**2))
        cost = np.minimum(grid[i, j], cost)
        grid[i, j] = cost
        
    def min_x(i, j):
        return grid[i, max(j-1, 0):min(j+2, nx)].min()
    
    def min_y(i, j):
        return grid[max(i-1, 0):min(i+2, ny), j].min()
    
    # Sweeps
    for _ in range(4):
        # Up-Right
        for i in range(0, ny):
            for j in range(0, nx):
                y_min = min_y(i, j)
                x_min = min_x(i, j)
                new_grid(i, j, y_min, x_min)
        # Up-Left
        for i in range(0, ny):
            for j in range(nx - 1, -1, -1):
                y_min = min_y(i, j)
                x_min = min_x(i, j)
                new_grid(i, j, y_min, x_min)
        # Down-Right
        for i in range(ny - 1, -1, -1):
            for j in range(0, nx):
                y_min = min_y(i, j)
                x_min = min_x(i, j)
                new_grid(i, j, y_min, x_min)
        # Down-Left
        for i in range(ny - 1, -1, -1):
            for j in range(nx - 1, -1, -1):
                y_min = min_y(i, j)
                x_min = min_x(i, j)
                new_grid(i, j, y_min, x_min)
    for _ in range(4):
        # Up-Right
        for j in range(0, nx):
            for i in range(0, ny):
                y_min = min_y(i, j)
                x_min = min_x(i, j)
                new_grid(i, j, y_min, x_min)
        # Up-Left
        for j in range(nx - 1, -1, -1):
            for i in range(0, ny):
                y_min = min_y(i, j)
                x_min = min_x(i, j)
                new_grid(i, j, y_min, x_min)
        # Down-Right
        for j in range(0, nx):
            for i in range(ny - 1, -1, -1):
                y_min = min_y(i, j)
                x_min = min_x(i, j)
                new_grid(i, j, y_min, x_min)
        # Down-Left
        for j in range(nx - 1, -1, -1):
            for i in range(ny - 1, -1, -1):
                y_min = min_y(i, j)
                x_min = min_x(i, j)
                new_grid(i, j, y_min, x_min)
    return grid
        
    
# Output image
def get_image(grid, obstacles, path, image_num):
	for x in obstacles:
		grid[x] = 0	
	grid = 150*grid/grid.max()
	for x in path:
		grid[x] = 200
	for x in obstacles:
		grid[x] = 255		
	img = Image.fromarray(np.uint8(grid))
	img.save(f"images/path" + str(image_num) + ".png")
	img.show()
 
def path_gen(grid, start, end, obstacles=None):
    path = [end]
    pos = end
    while pos != start:
        pos = select_dir(grid, pos, obstacles=obstacles)
        if pos is None:
            return None
        path.insert(0, pos)
    return path

def neighbors(x, shape, obstacles=None):
    if obstacles is None:
        obstacles = []
    ny, nx = shape
    i = x[0]
    j = x[1]
    N_ij = [
        (i - 1, j),
        (i + 1, j),
        (i, j - 1),
        (i, j + 1),
        (i - 1, j - 1),
        (i - 1, j + 1),
        (i + 1, j - 1),
        (i + 1, j + 1),
    ]
    good_neighbors = [
        (m, n)
        for (m, n) in N_ij
        if (m >= 0 and m < ny and n >= 0 and n < nx and (m, n) not in obstacles)
    ]
    return good_neighbors

def select_dir(grid, xk, obstacles=None):
    ny, nx = grid.shape
    i = xk[0]
    j = xk[1]
    loc_grids = neighbors(xk, grid.shape, obstacles=obstacles)
    loc_diff = [
        (grid[i, j] - grid[m, n]) / np.sqrt((i - m) ** 2 + (j - n) ** 2)
        for (m, n) in loc_grids
    ]
    loc_diff = np.maximum(0, loc_diff)
    if np.sum(loc_diff) > 0:
        loc_prob = loc_diff / np.sum(loc_diff)
        idx = np.random.choice(len(loc_prob), 1, p=loc_prob)[0]
        return loc_grids[idx]
    else:
        return None

def main():
    # obstacles represented as a large value
    obstacle_value = 1e10
    
    # distance between each grid point
    h = 1
    
    # creating the grid
    nx = 256
    ny = 256
    f = 1e-3*np.ones((ny, nx))
    
    # path1.png
    # image_num = 1
    # obstacles = [(i,j) for i, j in itertools.product(range(ny//4, 3*ny//4),range(nx//2 - 10, nx//2 + 10))]
    
    # path2.png
    # image_num = 2
    # obstacles = [(i,j) for i, j in itertools.product(range(2*ny//16, 11*ny//16),range(nx//2 - 10, nx//2 + 10))]
    
    # path3.png
    # image_num = 3
    # obstacles = [(i,j) for i, j in itertools.product(range(5*ny//16, 14*ny//16),range(nx//2 - 10, nx//2 + 10))]
    
    # path4.png
    # image_num = 4
    # obstacles = [(i,j) for i, j in itertools.product(range(ny//2 - 30, ny//2 + 30),range(3*nx//10, 7*nx//10))]
    
    # path5.png
    # image_num = 5
    # obstacles = [(i,j) for i, j in itertools.product(range(ny//2 - 10, ny//2 + 10),range(2*nx//10, 8*nx//10))]
    # obstacles += [(i,j) for i, j in itertools.product(range(ny//4, 3*ny//4),range(nx//2 - 5, nx//2 + 5))]
    
    # path6.png
    image_num = 6
    obstacles = [(i,j) for i, j in itertools.product(range(ny//2 - 10, ny//2 + 10),range(2*nx//15, 13*nx//15))]
    obstacles += [(i,j) for i, j in itertools.product(range(ny//4, 3*ny//4),range(nx//2 - 25, nx//2 - 15))]
    obstacles += [(i,j) for i, j in itertools.product(range(ny//4, 3*ny//4),range(nx//2 + 15, nx//2 + 25))]
    
    
    for x in obstacles:
        f[x] = obstacle_value
    
    start = (ny//2-1, 0)
    end = (ny//2-1, nx-1)
    grid = obstacle_value*np.ones((ny, nx))
    grid[start] = 0
    
    grid = fsm_2d(grid, f, h)
    
    path = path_gen(grid, start, end, obstacles=obstacles) 
    
    get_image(grid.copy(), obstacles, path, image_num)

if __name__ == "__main__":
    main()