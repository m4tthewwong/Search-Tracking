import numpy as np

def fsm_1d(size, h):
    # Inputs:
    #   1) size: size of the grid
    #   2) h: distance between each grid point
    # Outputs:
    #   1) grid: 1D array of grid points
    # -----------------------------------------------
    grid = np.full(size, 9999)
    grid[3] = grid[7] = 0
    for i in range(1, size-1):
        grid[i] = min(min(grid[i-1], grid[i+1]) + h, grid[i])
    print("After Sweep: ", grid)
    for j in range(size-2, -1, -1):
        grid[j] = min(min(grid[j-1], grid[j+1]) + h, grid[j])
    print("After Backtrack: ", grid)
 
def main():
    fsm_1d(10, 1)
 
if __name__ == '__main__':
    main()