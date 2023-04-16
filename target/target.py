import numpy as np
from FSM import fsm_2d
from PIL import Image

class Target (input):
    def __init__ (self, f, start, end, obstacles):
        self.f = f.copy()
        self.start = start
        self.end = end
        self.ny, self.nx = self.f.shape
        self.h = 1
        self.speed = 1
        self.pos = self.start
        self.path = [self.pos]
        self.obstacles = obstacles
        
    # start becomes where you are and the path is updated for a dynamic environment
    def forward(self, f, end):
        if f is not None:
            self.f = f
            self.grid[:] = 1e10
            self.grid[self.xk] = 0
            self.grid = fsm_2d(self.grid, self.f, self.h)
            self.end = end
        if f is not None or self.end is not None:
            p = fsm_2d.path_gen(self.grid, self.xk, self.end, self.obstacles) 
        else: 
            raise ValueError("either we are at the end or no path exists")
        
        return
            
def main():
    return
    
if __name__ == "__main__":
    main()
        