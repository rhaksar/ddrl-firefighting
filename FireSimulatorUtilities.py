from FireSimulator import *
import numpy as np

def CreateImage(state, position, dim=8):
  """
  helper function to create an image of a subset of the forest

  Inputs:
  - state: 2D numpy array containing full state of forest
  - position: numpy array of (x,y) position of agent
  - dim: output dimension of image

  Returns:
  - image: size (3,dim,dim) image of forest, padded with
           healthy trees if necessary
  """
  grid_size = state.shape[0]

  image = np.zeros((3,dim,dim)).astype(np.uint8)
  image[1,:,:] = 128 # initialize image as all healthy trees

  pos_round = int(np.rint(position))
  c = x_to_col(pos_rnd[0])
  r = y_to_row(grid_size,pos_rnd[1])
  half = dim//2

  for ri,dr in enumerate(np.arange(-half,half,1)):
    for ci,dc in enumerate(np.arange(-half,half,1)):
      rn = r + dr
      cn = c + dc
      if rn>=0 and rn<grid_size and cn>=0 and cn<grid_size:
        if state[rn,cn] == 0:
          image[:,ri,ci] = np.array([0,128,0])
        elif state[rn,cn] == 1:
          image[:,ri,ci] = np.array([128,0,0])
        else:
          image[:,ri,ci] = np.zeros((3))

  return image

def FindIntersections(state, waypoints):
  """
  helper function to find the intersection of a path with a grid

  Inputs:
  - state: 2D numpy array containing full state of forest
  - waypoints: numpy array of size (k,2) containing path waypoints

  Returns:
  - control: list of (x,y) positions of trees on fire that lie on
             the path defined by the waypoints
  """

  control = []
  grid_size = state.shape[0]
  k = waypoints.shape[0]
  delta = 0.2 # step increment on path

  x_glb = waypoints[:,0]
  y_glb = waypoints[:,1]

  # for each line segment:
  #   create intermediate points
  #   generate new coordinates
  #   check if new coordinate is a tree on fire
  #   if so, and it was not previously added, add to the control list
  
  for l in range(k-1):
    dist = np.sqrt((x_glb[l]-x_glb[l+1])**2 + (y_glb[l]-y_glb[l+1])**2)
    n = int(dist//delta)
    steps = np.linspace(0,1,n)

    for t in steps:
      new_x = (1-t)*x_glb[l] + t*x_glb[l+1]
      new_y = (1-t)*y_glb[l] + t*y_glb[l+1]

      new_x_int = int(np.rint(new_x))
      new_y_int = int(np.rint(new_y))

      c = x_to_col(new_x_int)
      r = y_to_row(grid_size,new_y_int)
      if r>=0 and r<grid_size and c>=0 and c<grid_size:
        if state[r,c] == 1:
          tree_x = col_to_x(c)
          tree_y = row_to_y(grid_size,r)
          if (tree_x,tree_y) not in control:
            control.append((tree_x,tree_y))
              
  return control