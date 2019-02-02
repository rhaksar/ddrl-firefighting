import numpy as np


def latticeforest_image(state, position, image_dims):
    """
    helper function to create an image of the forest.

    Inputs:
    - state: 2D numpy array containing full state of forest
    - position: numpy array of (row, col) position of where to take image
    - image_dims: output size (height, width) of image

    Returns:
    - image: 2D numpy array of size image_size representing
             image of forest, padded with healthy trees, if necessary
    """
    image = np.zeros(image_dims).astype(np.uint8)

    half_row = (image_dims[0]-1)//2
    half_col = (image_dims[1]-1)//2

    for ri, dr in enumerate(np.arange(-half_row, half_row+1, 1)):
        for ci, dc in enumerate(np.arange(-half_col, half_col+1, 1)):
            r = position[0] + dr
            c = position[1] + dc

            if 0 <= r < state.shape[0] and 0 <= c < state.shape[1]:
                image[ri, ci] = state[r, c]

    return image


def actions2trajectory(position, actions):
    """
    helper function to map a set of actions to a trajectory.

    action convention:
    1 - upper left, 2 - up,   3 - upper right
    4 - left,       0 - stop, 5 - right
    6 - lower left, 7 - down, 8 -lower right

    Inputs:
    - position: numpy array representing (x, y) position
    - actions: list of integers (see convention) describing movement actions

    Returns:
    - trajectory: list of (x, y) positions created by taking actions, where the
                  first element is the input position
    """

    trajectory = []
    x, y = position
    trajectory.append((x, y))
    for a in actions:
        x, y = trajectory[-1]
        if a == 0:
            trajectory.append((x, y))
        elif a == 1:
            trajectory.append((x-1, y+1))
        elif a == 2:
            trajectory.append((x, y+1))
        elif a == 3:
            trajectory.append((x+1, y+1))
        elif a == 4:
            trajectory.append((x-1, y))
        elif a == 5:
            trajectory.append((x+1, y))
        elif a == 6:
            trajectory.append((x-1, y-1))
        elif a == 7:
            trajectory.append((x, y-1))
        elif a == 8:
            trajectory.append((x+1, y-1))

    return trajectory


# helper functions to map (x, y) to (row, col) and vice versa
def rc2xy(height, rc):
    return rc[1]+1, height-rc[0]


def xy2rc(height, xy):
    return height-xy[1], xy[0]-1


def move_toward_center(agent):
    """
    helper function to determine action that moves agent closest to the fire ignition location.
    """
    distances = []
    for idx, a in enumerate([2, 5, 7, 4, 1, 3, 8, 6]):
        new_position = actions2trajectory(agent.position, [a])[1]
        incentive = -(8-idx)*0.1  # bias choice towards certain order of agent actions
        d = np.linalg.norm(agent.fire_center - np.asarray(new_position), 1) + incentive
        distances.append((d, a))

    _, action = min(distances, key=lambda t: t[0])
    return action


def reward(forest_state, agent):
    """
    agent reward function.
    """

    total_reward = 0

    x1, y1 = agent.position
    x2, y2 = agent.next_position

    # determine image cell corresponding to new position
    r_image = -y2 + y1 + agent.image_dims[1]//2
    c_image = x2 - x1 + agent.image_dims[0]//2

    r_forest, c_forest = xy2rc(forest_state.shape[1], agent.next_position)

    # if agent moved to a tree on fire, determine how many healthy neighbors it has
    # agent is reward for treating fires with more healthy neighbors
    if agent.image[r_image, c_image] == agent.on_fire:
        healthy_neighbors = 0
        for (dr, dc) in agent.fire_neighbors:
            rn, cn = r_forest + dr, c_forest + dc
            if rn < 0 or rn >= forest_state.shape[1] or cn < 0 or cn >= forest_state.shape[0]:
                healthy_neighbors += 1
            elif forest_state[rn, cn] == agent.healthy:
                healthy_neighbors += 1

        if healthy_neighbors > 0:
            total_reward += 1
        else:
            total_reward += -2

    # if the agent moved to a healthy tree, reward the move if it there are neighboring trees
    # that are either on fire or burnt
    elif agent.image[r_image, c_image] == agent.healthy:
        non_healthy_numbers = 0
        for (dr, dc) in agent.move_deltas:
            rn, cn = r_forest + dr, c_forest + dc
            if rn < 0 or rn >= forest_state.shape[1] or cn < 0 or cn >= forest_state.shape[0]:
                continue
            elif forest_state[rn, cn] in [agent.on_fire, agent.burnt]:
                non_healthy_numbers += 1

        if non_healthy_numbers > 0:
            total_reward += 0.5
        else:
            total_reward += -1

    # if the agent is responsible for avoiding the closest other agent, penalize it for moving too close
    # and reward it for moving away
    if agent.numeric_id > agent.closest_agent_id:
        if np.linalg.norm(agent.next_position - agent.closest_agent_position, 2) <= 1:
            total_reward += -10
        elif np.linalg.norm(agent.position - agent.closest_agent_position, 2) <= 1 \
                < np.linalg.norm(agent.next_position - agent.closest_agent_position, 2):
            total_reward += 1

    # reward actions that are moving the agent clockwise around the fire ignition location
    move_vector = agent.next_position - agent.position
    norm = np.linalg.norm(move_vector, 2)
    if norm != 0:
        move_vector = move_vector / norm

    center_vector = agent.position - agent.fire_center
    norm = np.linalg.norm(center_vector, 2)
    if norm != 0:
        center_vector = center_vector / norm

    rotation_score = -1*np.cross(center_vector, move_vector)
    if rotation_score >= 0:
        total_reward += 1

    return total_reward


def heuristic(agent):
    """
    hand-tuned heuristic to generate single action for an agent.
    """
    action = None

    # if agent reached fire, move on the fire boundary to apply retardant
    if agent.reached_fire:
        # find the action that is closest to simply rotating clockwise about the fire center
        distances = []
        fire_center_vector = agent.position - agent.fire_center
        norm = np.linalg.norm(fire_center_vector, 2)
        if norm != 0:
            fire_center_vector = fire_center_vector / norm

        for a in range(1, 9):
            new_position = actions2trajectory(agent.position, [a])[1]
            move_vector = np.asarray(new_position) - agent.position
            move_vector = move_vector / np.linalg.norm(move_vector, 2)
            distances.append((np.cross(fire_center_vector, move_vector), new_position, a))

        _, circular_position, action = min(distances, key=lambda t: t[0])

        # calculate corresponding location in image
        ri = -circular_position[1] + agent.position[1] + (agent.image_dims[0]-1)//2
        ci = circular_position[0] - agent.position[0] + (agent.image_dims[1]-1)//2

        # determine "left" and "right" actions, relative to the rotation action
        left_action = None
        right_action = None
        if action == 1:
            left_action = [6, 4]
            right_action = 2
        elif action == 2:
            left_action = [4, 1]
            right_action = 3
        elif action == 3:
            left_action = [1, 2]
            right_action = 5
        elif action == 5:
            left_action = [2, 3]
            right_action = 8
        elif action == 8:
            left_action = [3, 5]
            right_action = 7
        elif action == 7:
            left_action = [5, 8]
            right_action = 6
        elif action == 6:
            left_action = [8, 7]
            right_action = 4
        elif action == 4:
            left_action = [7, 6]
            right_action = 1

        # determine if there is a "left" action that moves the agent to a tree on fire or a burnt tree
        move_left = False
        for a in left_action:
            new_position = actions2trajectory(agent.position, [a])[1]
            ro = -new_position[1] + agent.position[1] + (agent.image_dims[0]-1)//2
            co = new_position[0] - agent.position[0] + (agent.image_dims[1]-1)//2
            if agent.image[ro, co] == agent.on_fire:
                circular_position = new_position
                action = a
                move_left = True
                break

        if not move_left:
            for a in left_action:
                new_position = actions2trajectory(agent.position, [a])[1]
                ro = -new_position[1] + agent.position[1] + (agent.image_dims[0]-1)//2
                co = new_position[0] - agent.position[0] + (agent.image_dims[1]-1)//2
                if agent.image[ro, co] == agent.burnt:
                    circular_position = new_position
                    action = a
                    move_left = True
                    break

        # if not moving left, check if proposed action leads to moving too far away from the fire boundary:
        # check if the agent is moving to a healthy tree with many healthy neighboring trees
        if not move_left and agent.image[ri, ci] == agent.healthy:
            healthy_neighbors = 0
            for (dr, dc) in agent.move_deltas:
                rn, cn = ri + dr, ci + dc
                # assume neighbors out of image are healthy
                if 0 <= rn < agent.image_dims[1] and 0 <= cn < agent.image_dims[0]:
                    if agent.image[rn, cn] == agent.healthy:
                        healthy_neighbors += 1
                else:
                    healthy_neighbors += 1

            if healthy_neighbors >= 6:
                circular_position = actions2trajectory(agent.position, [right_action])[1]
                action = right_action

        # if the proposed action is too close to the closest agent, stop
        if np.linalg.norm(circular_position - agent.closest_agent_position, 2) <= 1 \
                and agent.numeric_id > agent.closest_agent_id:
            action = 0

    # move towards the fire ignition center
    else:
        action = move_toward_center(agent)

    return action
