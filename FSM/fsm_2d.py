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
def get_image(grid, obstacles, path):
	for x in obstacles:
		grid[x] = 0	
	grid = 150*grid/grid.max()
	for x in path:
		grid[x] = 200
	for x in obstacles:
		grid[x] = 255		
	img = Image.fromarray(np.uint8(grid))
	img.save(f"images/img.png")
	img.show()
 
def path_gen(u, gamma, omega, obstacles=None, act_type="relu", rnd=True):
    path = [omega]
    pos = omega
    while pos != gamma:
        pos = select_dir(u, pos, obstacles=obstacles, act_type=act_type, rnd=rnd)
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

def select_dir(u, xk, obstacles=None, act_type="relu", rnd=True):
    ny, nx = u.shape
    i = xk[0]
    j = xk[1]
    loc_grids = neighbors(xk, u.shape, obstacles=obstacles)
    loc_diff = [
        (u[i, j] - u[m, n]) / np.sqrt((i - m) ** 2 + (j - n) ** 2)
        for (m, n) in loc_grids
    ]
    loc_diff = local_activation(act_type=act_type)(loc_diff)
    if np.sum(loc_diff) > 0:
        loc_prob = loc_diff / np.sum(loc_diff)
        if rnd:
            idx = np.random.choice(len(loc_prob), 1, p=loc_prob)[0]
        else:
            idx = np.argmax(loc_prob)
        return loc_grids[idx]
    else:
        return None

def local_activation(act_type="relu"):
    def relu(x):
        return np.maximum(0, x)
    def relu_a(x,a=-1e-3):
        return np.maximum(a, x) - a

    if act_type == "relu":
        return relu
    else:
        return relu_a

def main():
    obstacle_value = 1e10
    h = 1 # distance between each grid point
    
    #FSM 2D
    nx = 256
    ny = 256
    f = 1e-3*np.ones((ny, nx))
    obstacles = [(i,j) for i, j in itertools.product(range(ny//4, 3*ny//4),range(nx//2 - 10, nx//2 + 10))]
    
    for x in obstacles:
        f[x] = obstacle_value
    
    gamma = (ny//2-1, 0)
    omega = (ny//2-1, nx-1)
    grid = obstacle_value*np.ones((ny, nx))
    grid[gamma] = 0
    
    grid = fsm_2d(grid, f, h)
    
    act = 'relu_a'
    rnd = True
    
    path = path_gen(grid, gamma, omega, obstacles=obstacles, act_type=act, rnd=rnd) 
    
    get_image(grid.copy(), obstacles, path)

if __name__ == "__main__":
    main()