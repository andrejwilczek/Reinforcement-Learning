import numpy as np
import matplotlib.pyplot as plt
import time
from IPython import display
import math

# Andrej Wilzcek 880707-7477
# Kildo Alias 971106-7430


# Implemented methods
methods = ['DynProg', 'ValIter']

# Some colours
LIGHT_RED = '#FFC4CC'
LIGHT_GREEN = '#95FD99'
BLACK = '#000000'
WHITE = '#FFFFFF'
LIGHT_PURPLE = '#E8D0FF'
LIGHT_ORANGE = '#FAE0C3'


class Maze:

    # Actions
    # STAY = 0
    # MOVE_LEFT = 1
    # MOVE_RIGHT = 2
    # MOVE_UP = 3
    # MOVE_DOWN = 4
    STAY = 4
    MOVE_LEFT = 0
    MOVE_RIGHT = 1
    MOVE_UP = 2
    MOVE_DOWN = 3

    # Give names to actions
    actions_names = {
        STAY: "stay",
        MOVE_LEFT: "move left",
        MOVE_RIGHT: "move right",
        MOVE_UP: "move up",
        MOVE_DOWN: "move down"
    }

    # Reward values
    STEP_REWARD = 0
    GOAL_REWARD = 10
    IMPOSSIBLE_REWARD = -math.inf
    CAUGHT_REWARD = -50

    def __init__(self, maze, weights=None, random_rewards=False, start=(0, 0, 1, 2)):
        """ Constructor of the environment Maze.
        """
        self.maze = maze
        self.start = start
        self.actions = self.__actions()
        self.states, self.map = self.__states()
        self.policeActions = self.__police_actions()
        self.n_policeActions = len(self.policeActions)
        self.n_actions = len(self.actions)
        self.n_states = len(self.states)
        self.transition_probabilities = self.__transitions()
        self.rewards = self.__rewards(weights=weights,
                                      random_rewards=random_rewards)

    def __actions(self):
        actions = dict()
        actions[self.STAY] = (0, 0)
        actions[self.MOVE_LEFT] = (0, -1)
        actions[self.MOVE_RIGHT] = (0, 1)
        actions[self.MOVE_UP] = (-1, 0)
        actions[self.MOVE_DOWN] = (1, 0)
        return actions

    def __police_actions(self):
        policeActions = dict()
        policeActions[self.MOVE_LEFT] = (0, -1)
        policeActions[self.MOVE_RIGHT] = (0, 1)
        policeActions[self.MOVE_UP] = (-1, 0)
        policeActions[self.MOVE_DOWN] = (1, 0)
        return policeActions

    def __states(self):
        states = dict()
        map = dict()
        end = False
        s = 0
        for i in range(self.maze.shape[0]):
            for j in range(self.maze.shape[1]):
                if self.maze[i, j] != 1:
                    for k in range(self.maze.shape[0]):
                        for l in range(self.maze.shape[1]):
                            states[s] = (i, j, k, l)
                            map[(i, j, k, l)] = s
                            s += 1

        return states, map

    def __move(self, state, action, policeAction):
        """ Makes a step in the maze, given a current position and an action.
            If the action STAY or an inadmissible action is used, the agent stays in place.

            :return tuple next_cell: Position (x,y) on the maze that agent transitions to.
        """
        # Has the police caught the robber?
        if self.states[state][0] == self.states[state][2] and self.states[state][1] == self.states[state][3]:
            return self.map[self.start]

        # Compute the future minotaur position given current (state, action)
        policeRow = self.states[state][2] + self.policeActions[policeAction][0]
        policeCol = self.states[state][3] + self.policeActions[policeAction][1]

        # Is the future position an impossible one ?
        outside = (policeRow == -1) or (policeRow == self.maze.shape[0]) or \
            (policeCol == -1) or (policeCol ==
                                  self.maze.shape[1])

        if outside:
            return None

        # Compute the future player position given current (state, action)
        row = self.states[state][0] + self.actions[action][0]
        col = self.states[state][1] + self.actions[action][1]

        # Is the future position an impossible one ?
        hitting_maze_walls = (row == -1) or (row == self.maze.shape[0]) or \
            (col == -1) or (col == self.maze.shape[1]) or \
            (self.maze[row, col] == 1)

        # Based on the impossiblity check return the next state.
        if hitting_maze_walls:
            return self.map[(self.states[state][0], self.states[state][1], policeRow, policeCol)]
        else:
            return self.map[(row, col, policeRow, policeCol)]

    def __transitions(self):
        """ Computes the transition probabilities for every state action pair.
            :return numpy.tensor transition probabilities: tensor of transition
            probabilities of dimension S*S*A
        """
        # Initialize the transition probailities tensor (S,S,A)
        dimensions = (self.n_states, self.n_states, self.n_actions)
        transition_probabilities = np.zeros(dimensions)

        # Compute the transition probabilities. Note that the transitions
        # are deterministic.
        for s in range(self.n_states):
            for a in range(self.n_actions):
                nextStates = []

                # Check police actions
                deltaX = self.states[s][0]-self.states[s][2]
                deltaY = self.states[s][1]-self.states[s][3]

                if deltaX == 0 and deltaY < 0:
                    policeList = [0, 2, 3]
                elif deltaX == 0 and deltaY > 0:
                    policeList = [1, 2, 3]
                elif deltaY == 0 and deltaX < 0:
                    policeList = [0, 1, 2]
                elif deltaY == 0 and deltaX > 0:
                    policeList = [0, 1, 3]
                elif deltaX < 0 and deltaY < 0:
                    policeList = [0, 2]
                elif deltaX > 0 and deltaY > 0:
                    policeList = [1, 3]
                elif deltaX > 0 and deltaY < 0:
                    policeList = [0, 3]
                elif deltaX < 0 and deltaY > 0:
                    policeList = [1, 2]
                else:
                    policeList = [0]

                for pa in policeList:
                    next_s = self.__move(s, a, pa)
                    if next_s == self.map[self.start]:
                        nextStates.append(next_s)
                        break
                    if next_s != None:
                        nextStates.append(next_s)
                prob = 1.0/len(nextStates)
                for next_s in nextStates:
                    transition_probabilities[next_s, s, a] = prob
        return transition_probabilities

    def __rewards(self, weights=None, random_rewards=None):

        rewards = np.zeros((self.n_states, self.n_actions))

        # If the rewards are not described by a weight matrix

        for s in range(self.n_states):
            for a in range(self.n_actions):
                nextStates = []
                # Check police actions
                deltaX = self.states[s][0]-self.states[s][2]
                deltaY = self.states[s][1]-self.states[s][3]

                if deltaX == 0 and deltaY < 0:
                    policeList = [0, 2, 3]
                elif deltaX == 0 and deltaY > 0:
                    policeList = [1, 2, 3]
                elif deltaY == 0 and deltaX < 0:
                    policeList = [0, 1, 2]
                elif deltaY == 0 and deltaX > 0:
                    policeList = [0, 1, 3]
                elif deltaX < 0 and deltaY < 0:
                    policeList = [0, 2]
                elif deltaX > 0 and deltaY > 0:
                    policeList = [1, 3]
                elif deltaX > 0 and deltaY < 0:
                    policeList = [0, 3]
                elif deltaX < 0 and deltaY > 0:
                    policeList = [1, 2]
                else:
                    policeList = [0]

                for pa in policeList:
                    next_s = self.__move(s, a, pa)
                    if next_s != None:
                        nextStates.append(next_s)

                for next_s in nextStates:
                    # Reward for restarting
                    if self.states[s][0] == self.states[s][2] and self.states[s][1] == self.states[s][3] and next_s == self.map[self.start]:
                        rewards[s, a] += 0
                     # Rewrd for hitting a wall
                    elif self.states[s][0] == self.states[next_s][0] and self.states[s][1] == self.states[next_s][1] and a != self.STAY:
                        rewards[s, a] += self.IMPOSSIBLE_REWARD
                    # reward for being caught
                    elif self.states[next_s][0] == self.states[next_s][2] and self.states[next_s][1] == self.states[next_s][3]:
                        rewards[s, a] += self.CAUGHT_REWARD
                    # Reward for reaching the exit
                    elif self.states[s][0] == self.states[next_s][0] and self.states[s][1] == self.states[next_s][1] and self.maze[self.states[next_s][0], self.states[next_s][1]] == 2:
                        rewards[s, a] += self.GOAL_REWARD

# self.states[s][0] == self.states[next_s][0] and self.states[s][1] == self.states[next_s][1] and
                rewards[s, a] /= len(nextStates)
        return rewards

    def simulate(self, start, policy, method):
        if method not in methods:
            error = 'ERROR: the argument method must be in {}'.format(methods)
            raise NameError(error)

        path = list()
        if method == 'ValIter':
            # Initialize current state, next state and time
            t = 1
            s = self.map[start]
            # Add the starting position in the maze to the path
            path.append(start)

            # Check police actions
            deltaX = self.states[s][0]-self.states[s][2]
            deltaY = self.states[s][1]-self.states[s][3]

            if deltaX == 0 and deltaY < 0:
                policeList = [0, 2, 3]
            elif deltaX == 0 and deltaY > 0:
                policeList = [1, 2, 3]
            elif deltaY == 0 and deltaX < 0:
                policeList = [0, 1, 2]
            elif deltaY == 0 and deltaX > 0:
                policeList = [0, 1, 3]
            elif deltaX < 0 and deltaY < 0:
                policeList = [0, 2]
            elif deltaX > 0 and deltaY > 0:
                policeList = [1, 3]
            elif deltaX > 0 and deltaY < 0:
                policeList = [0, 3]
            elif deltaX < 0 and deltaY > 0:
                policeList = [1, 2]
            else:
                policeList = [0]

            # calculate a random move for the Minotaur
            while True:
                randomMove = np.random.randint(0, len(policeList))
                randomMove = policeList[randomMove]
                # Compute the future minotaur position given current (state, action)
                policeRow = self.states[s][2] + \
                    self.policeActions[randomMove][0]
                policeCol = self.states[s][3] + \
                    self.policeActions[randomMove][1]
                # Is the future position an impossible one ?
                outside = (policeRow == -1) or (policeRow == self.maze.shape[0]) or \
                    (policeCol == -1) or (policeCol ==
                                          self.maze.shape[1])
                if not outside:
                    break

            # Move to next state given the policy and the current state
            next_s = self.__move(s, policy[s], randomMove)

            # Add the position in the maze corresponding to the next state
            # to the path
            path.append(self.states[next_s])
            # Loop while state is not the goal state
            T = 0
            while T < 100:
                # Update state
                s = next_s
                deltaX = self.states[s][0]-self.states[s][2]
                deltaY = self.states[s][1]-self.states[s][3]

                if deltaX == 0 and deltaY < 0:
                    policeList = [0, 2, 3]
                elif deltaX == 0 and deltaY > 0:
                    policeList = [1, 2, 3]
                elif deltaY == 0 and deltaX < 0:
                    policeList = [0, 1, 2]
                elif deltaY == 0 and deltaX > 0:
                    policeList = [0, 1, 3]
                elif deltaX < 0 and deltaY < 0:
                    policeList = [0, 2]
                elif deltaX > 0 and deltaY > 0:
                    policeList = [1, 3]
                elif deltaX > 0 and deltaY < 0:
                    policeList = [0, 3]
                elif deltaX < 0 and deltaY > 0:
                    policeList = [1, 2]
                else:
                    policeList = [0]

                while True:
                    randomMove = np.random.randint(0, len(policeList))
                    randomMove = policeList[randomMove]

                    # Compute the future minotaur position given current (state, action)
                    policeRow = self.states[s][2] + \
                        self.policeActions[randomMove][0]
                    policeCol = self.states[s][3] + \
                        self.policeActions[randomMove][1]

                    # Is the future position an impossible one ?
                    outside = (policeRow == -1) or (policeRow == self.maze.shape[0]) or \
                        (policeCol == -1) or (policeCol ==
                                              self.maze.shape[1])

                    if not outside:
                        break

                # Move to next state given the policy and the current state
                next_s = self.__move(s, policy[s], randomMove)
                # Add the position in the maze corresponding to the next state
                # to the path
                path.append(self.states[next_s])
                # Update time and state for next iteration
                t += 1
                T += 1
        print(path)
        return path

    def show(self):
        print('The states are :')
        print(self.states)
        print('The actions are:')
        print(self.actions)
        print('The mapping of the states:')
        print(self.map)
        print('The rewards:')
        print(self.rewards[8, :])


def dynamic_programming(env, horizon):
    """ Solves the shortest path problem using dynamic programming
        :input Maze env           : The maze environment in which we seek to
                                    find the shortest path.
        :input int horizon        : The time T up to which we solve the problem.
        :return numpy.array V     : Optimal values for every state at every
                                    time, dimension S*T
        :return numpy.array policy: Optimal time-varying policy at every state,
                                    dimension S*T
    """

    # The dynamic prgramming requires the knowledge of :
    # - Transition probabilities
    # - Rewards
    # - State space
    # - Action space
    # - The finite horizon
    p = env.transition_probabilities
    r = env.rewards
    n_states = env.n_states
    n_actions = env.n_actions
    T = horizon

    # The variables involved in the dynamic programming backwards recursions
    V = np.zeros((n_states, T+1))
    policy = np.zeros((n_states, T+1))
    Q = np.zeros((n_states, n_actions))

    # Initialization
    Q = np.copy(r)
    V[:, T] = np.max(Q, 1)
    policy[:, T] = np.argmax(Q, 1)

    # The dynamic programming bakwards recursion
    for t in range(T-1, -1, -1):
        # Update the value function acccording to the bellman equation
        for s in range(n_states):
            for a in range(n_actions):
                # Update of the temporary Q values
                Q[s, a] = r[s, a] + np.dot(p[:, s, a], V[:, t+1])
        # Update by taking the maximum Q value w.r.t the action a
        V[:, t] = np.max(Q, 1)
        # The optimal action is the one that maximizes the Q function
        policy[:, t] = np.argmax(Q, 1)
    return V, policy


def value_iteration(env, gamma, epsilon):
    """ Solves the shortest path problem using value iteration
        :input Maze env           : The maze environment in which we seek to
                                    find the shortest path.
        :input float gamma        : The discount factor.
        :input float epsilon      : accuracy of the value iteration procedure.
        :return numpy.array V     : Optimal values for every state at every
                                    time, dimension S*T
        :return numpy.array policy: Optimal time-varying policy at every state,
                                    dimension S*T
    """
    # The value itearation algorithm requires the knowledge of :
    # - Transition probabilities
    # - Rewards
    # - State space
    # - Action space
    # - The finite horizon
    p = env.transition_probabilities
    r = env.rewards
    n_states = env.n_states
    n_actions = env.n_actions

    # Required variables and temporary ones for the VI to run
    V = np.zeros(n_states)
    Q = np.zeros((n_states, n_actions))
    BV = np.zeros(n_states)
    # Iteration counter
    n = 0
    # Tolerance error
    tol = (1 - gamma) * epsilon/gamma

    # Initialization of the VI
    for s in range(n_states):
        for a in range(n_actions):
            Q[s, a] = r[s, a] + gamma*np.dot(p[:, s, a], V)
    BV = np.max(Q, 1)

    # Iterate until convergence
    while np.linalg.norm(V - BV) >= tol and n < 200:
        # Increment by one the numbers of iteration
        n += 1
        # Update the value function
        V = np.copy(BV)
        # Compute the new BV
        for s in range(n_states):
            for a in range(n_actions):
                Q[s, a] = r[s, a] + gamma*np.dot(p[:, s, a], V)
        BV = np.max(Q, 1)
        # Show error
        # print(np.linalg.norm(V - BV))

    # Compute policy
    policy = np.argmax(Q, 1)
    # Return the obtained policy
    return V, policy


def draw_maze(maze):

    # Map a color to each cell in the maze
    col_map = {0: WHITE, 1: BLACK,
               2: LIGHT_GREEN, -6: LIGHT_RED, -1: LIGHT_RED}

    # Give a color to each cell
    rows, cols = maze.shape
    colored_maze = [[col_map[maze[j, i]]
                     for i in range(cols)] for j in range(rows)]

    # Create figure of the size of the maze
    fig = plt.figure(1, figsize=(cols, rows))

    # Remove the axis ticks and add title title
    ax = plt.gca()
    ax.set_title('The Maze')
    ax.set_xticks([])
    ax.set_yticks([])

    # Give a color to each cell
    rows, cols = maze.shape
    colored_maze = [[col_map[maze[j, i]]
                     for i in range(cols)] for j in range(rows)]

    # Create figure of the size of the maze
    fig = plt.figure(1, figsize=(cols, rows))

    # Create a table to color
    grid = plt.table(cellText=None,
                     cellColours=colored_maze,
                     cellLoc='center',
                     loc=(0, 0),
                     edges='closed')
    # Modify the hight and width of the cells in the table
    tc = grid.properties()['children']
    for cell in tc:
        cell.set_height(1.0/rows)
        cell.set_width(1.0/cols)
    display.display(fig)
    display.clear_output(wait=True)
    # time.sleep(1)
    plt.show()


def animate_solution(maze, path):

    # Map a color to each cell in the maze
    col_map = {0: WHITE, 1: BLACK,
               2: LIGHT_GREEN, -6: LIGHT_RED, -1: LIGHT_RED}

    # Size of the maze
    rows, cols = maze.shape

    # Create figure of the size of the maze
    fig = plt.figure(1, figsize=(cols, rows))

    # Remove the axis ticks and add title title
    ax = plt.gca()
    ax.set_title('Policy simulation')
    ax.set_xticks([])
    ax.set_yticks([])

    # Give a color to each cell
    colored_maze = [[col_map[maze[j, i]]
                     for i in range(cols)] for j in range(rows)]

    # Create figure of the size of the maze
    fig = plt.figure(1, figsize=(cols, rows))

    # Create a table to color
    grid = plt.table(cellText=None,
                     cellColours=colored_maze,
                     cellLoc='center',
                     loc=(0, 0),
                     edges='closed')

    # Modify the hight and width of the cells in the table
    tc = grid.properties()['children']
    for cell in tc:
        cell.set_height(1.0/rows)
        cell.set_width(1.0/cols)

    # Update the color at each frame
    for i in range(len(path)):
        if i > 0:
            if path[i][0] == path[i][2] and path[i][1] == path[i][3]:
                grid.get_celld()[(path[i][2], path[i][3])
                                 ].set_facecolor(LIGHT_RED)
                grid.get_celld()[(path[i][2], path[i][3])].get_text().set_text(
                    'Player is eaten')

            grid.get_celld()[(path[i-1][0], path[i-1][1])
                             ].set_facecolor(col_map[maze[path[i-1][0], path[i-1][1]]])
            grid.get_celld()[(path[i-1][0], path[i-1][1])
                             ].get_text().set_text('')

            grid.get_celld()[(path[i-1][2], path[i-1][3])
                             ].set_facecolor(col_map[maze[path[i-1][2], path[i-1][3]]])
            grid.get_celld()[(path[i-1][2], path[i-1][3])
                             ].get_text().set_text('')

            grid.get_celld()[(path[i][0], path[i][1])
                             ].set_facecolor(LIGHT_ORANGE)
            grid.get_celld()[(path[i][0], path[i][1])
                             ].get_text().set_text('Robert')

            grid.get_celld()[(path[i][2], path[i][3])].set_facecolor(LIGHT_RED)
            grid.get_celld()[(path[i][2], path[i][3])
                             ].get_text().set_text('Snutn')

            playerLast = grid.get_celld()[(path[i-1][0], path[i-1][1])].xy
            playerNow = grid.get_celld()[(path[i][0], path[i][1])].xy
            deltaX = playerNow[0]-playerLast[0]
            deltaY = playerNow[1]-playerLast[1]

            plt.arrow(playerLast[0]+0.08, playerLast[1]+0.15,
                      deltaX, deltaY, width=0.005)

            minoLast = grid.get_celld()[(path[i-1][2], path[i-1][3])].xy
            minoNow = grid.get_celld()[(path[i][2], path[i][3])].xy
            deltaX = minoNow[0]-minoLast[0]
            deltaY = minoNow[1]-minoLast[1]

            plt.arrow(minoLast[0]+0.04, minoLast[1]+0.10,
                      deltaX, deltaY, width=0.005, color='red')

            if path[i][0] == path[i-1][0] and path[i][1] == path[i-1][1] and maze[(path[i][0], path[i][1])] == 2:
                print('Vi är i mål')
                grid.get_celld()[(path[i][0], path[i][1])
                                 ].set_facecolor(LIGHT_PURPLE)
                grid.get_celld()[(path[i][0], path[i][1])].get_text().set_text(
                    'Robert Is Robbing')

        else:
            grid.get_celld()[(path[i][0], path[i][1])
                             ].set_facecolor(LIGHT_ORANGE)
            grid.get_celld()[(path[i][0], path[i][1])
                             ].get_text().set_text('Robert')

            grid.get_celld()[(path[i][2], path[i][3])].set_facecolor(LIGHT_RED)
            grid.get_celld()[(path[i][2], path[i][3])
                             ].get_text().set_text('Snutn')

        display.display(fig)
        display.clear_output(wait=True)
        # time.sleep(1)
        plt.draw()
        plt.pause(0.1)
    display.display(fig)
    display.clear_output(wait=True)
    # time.sleep(1)
    plt.show()
