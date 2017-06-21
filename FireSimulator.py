import itertools
import math
import numpy as np

class FireSimulator(object):
  """
  A simulation of a forest fire using a discrete probabilistic grid model
  The model is adapted from literature:
    A. Somanath, S. Karaman, and K. Youcef-Toumi, "Controlling stochastic
    growth processes on lattices: Wildfire management with robotic fire
    extinguishers," in 53rd IEEE Conference on Decision and Control.
    IEEE, 2014, pp. 1432-1327

  """
  def __init__(self, grid_size, rng=None, fire_init=None, 
               alpha=0.2763, beta=np.exp(-1/10)):
    """
    Initializes a simulation object.
    Each tree in the grid model has three states: 
      0 - healthy tree
      1 - burning tree
      2 - burnt tree

    Inputs:
    - grid_size: size of forest, assumes square grid (minimum size: 4)
    - rng: random number generator seed to use for deterministic sampling
    - fire_init: list of tuples of (x,y) coordinates describing 
                 positions of initial fires
    - alpha: fire propagation parameter
    - beta: fire persistence parameter
    """

    self.grid_size = grid_size
    self.alpha = alpha
    self.beta = beta

    self.state = np.zeros((grid_size,grid_size)).astype(np.uint8)

    self.stats = np.zeros(3).astype(np.uint32)
    self.stats[0] += grid_size**2

    self.hlth_state = 0
    self.burn_state = 1
    self.dead_state = 2

    if rng is not None:
      np.random.seed(rng)

    self.iter = 0
    self.fires = []
    if fire_init is not None:
      self.fires = fire_init
      self.iter += 1
      for (x,y) in fire_init:
        c = x_to_col(x)
        r = y_to_row(grid_size,y)
        self.state[r,c] = 1

      self.stats[0] -= len(fire_init)
      self.stats[1] += len(fire_init)
    
    self.end = False

  def _persists(self, fire, action, dbeta):
    """
    Helper function to determine if a fire should continue to burn. Also
    defines how the action affects the state update.

    Inputs:
    - fire: tuple of (x,y) position 
    - action: list of trees that should be treated at current time step
    - dbeta: reduction in fire persistence parameter if tree is treated

    Returns:
      True if fire continues to burn
      False if fire burns out
    """

    delta = dbeta if fire in action else 0
    sample = np.random.rand()
    if sample > (self.beta - delta):
      x,y = fire
      c = x_to_col(x)
      r = y_to_row(self.grid_size,y)
      self.state[r,c] = 2
      self.stats[1] -= 1
      self.stats[2] += 1
      return False
    return True

  def step(self, action, dbeta=0):
    """
    Function to update the state of the forest fire

    Inputs:
    - action: list of tuples of (x,y) coordinates describing treatment location
    - dbeta: reduction in fire persistent parameter due to treatment
    """

    if self.end:
      print("process has terminated")
      return

    grid_size = self.grid_size

    # start a square of fires at center at iter 0
    if self.iter == 0:
      x = math.ceil(grid_size/2)
      deltas = [k for k in range(-1,3)]
      neighbors = itertools.product(deltas,deltas)

      for (dx,dy) in neighbors:
        xn = x + dx
        yn = x + dy

        cn = x_to_col(xn)
        rn = y_to_row(grid_size,yn)

        self.fires.append((xn,yn))
        self.state[rn,cn] = 1
        self.stats[0] -= 1
        self.stats[1] += 1

      self.iter += 1
      return

    neighbors = [(-1,0),(1,0),(0,-1),(0,1)]

    add = [] # list of trees that will catch on fire
    checked = [] # list of healthy trees that have been simulated
    num_fires = len(self.fires)

    # fire spread step
    # iterate over fires, find their healthy neighbors, and sample to 
    # determine if they catch fire
    for (x,y) in self.fires:

      for (dx,dy) in neighbors:
        xn = x + dx
        yn = y + dy

        if xn>=1 and xn<=grid_size and yn>=1 and yn<=grid_size:
          cn = x_to_col(xn)
          rn = y_to_row(grid_size,yn)

          if self.state[rn,cn] == self.hlth_state and (xn,yn) not in checked:
            num_neighbor_fires = 0

            for (dx2,dy2) in neighbors:
              xn2 = xn + dx2
              yn2 = yn + dy2

              if xn2>=1 and xn2<=grid_size and yn2>=1 and yn2<=grid_size:
                cn2 = x_to_col(xn2)
                rn2 = y_to_row(grid_size,yn2)

                if self.state[rn2,cn2] == self.burn_state:
                  num_neighbor_fires += 1

            prob = 1 - (1 - self.alpha)**num_neighbor_fires
            if np.random.rand() <= prob:
              add.append((xn,yn))

            checked.append((xn,yn))

    # fire burn out step [includes action]
    self.fires = [(x,y) for (x,y) in self.fires if self._persists((x,y),action,dbeta)]

    # update list of current fires
    for (x,y) in add:
      c = x_to_col(x)
      r = y_to_row(grid_size,y)
      self.fires.append((x,y))
      self.state[r,c] = 1

    self.stats[0] -= len(add)
    self.stats[1] += len(add)

    # terminate if no fires
    if not self.fires:
      self.iter += 1
      self.end = True
      return

    self.iter += 1
    return

# helper functions to map between (row,col) and (x,y)
def col_to_x(col):
  return col+1

def x_to_col(x):
  return x-1

def row_to_y(grid_size,row):
  return grid_size-row

def y_to_row(grid_size,y):
  return grid_size-y
