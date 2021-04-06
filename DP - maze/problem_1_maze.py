import numpy as np
import matplotlib.pyplot as plt
import time
from IPython import display

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
    STEP_REWARD = -1
    GOAL_REWARD = 0
    IMPOSSIBLE_REWARD = -100

    def __init__(self, maze, weights=None, random_rewards=False, minoStay=False):
        """ Constructor of the environment Maze.
        """
        self.maze = maze
        self.minoStay = minoStay
        self.actions = self.__actions()
        self.minoActions = self.__mino_actions()
        self.states, self.map = self.__states()
        self.n_minoActions = len(self.minoActions)
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

    def __mino_actions(self):
        minoActions = dict()
        minoActions[self.MOVE_LEFT] = (0, -1)
        minoActions[self.MOVE_RIGHT] = (0, 1)
        minoActions[self.MOVE_UP] = (-1, 0)
        minoActions[self.MOVE_DOWN] = (1, 0)
        if self.minoStay:
            minoActions[self.STAY] = (0, 0)
        return minoActions

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

    def __move(self, state, action, minoAction):
        """ Makes a step in the maze, given a current position and an action.
            If the action STAY or an inadmissible action is used, the agent stays in place.

            :return tuple next_cell: Position (x,y) on the maze that agent transitions to.
        """
        # Compute the future minotaur position given current (state, action)
        minoRow = self.states[state][2] + self.minoActions[minoAction][0]
        minoCol = self.states[state][3] + self.minoActions[minoAction][1]

        try:
            if (self.maze[minoRow, minoCol] == 1):
                minoRow += self.minoActions[minoAction][0]
                minoCol += self.minoActions[minoAction][1]
        except:
            return None

        # Is the future position an impossible one ?
        outside = (minoRow == -1) or (minoRow == self.maze.shape[0]) or \
            (minoCol == -1) or (minoCol ==
                                self.maze.shape[1]) or (self.maze[minoRow, minoCol] == 1)

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
            return self.map[(self.states[state][0], self.states[state][1], minoRow, minoCol)]
        else:
            return self.map[(row, col, minoRow, minoCol)]

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
                for ma in range(self.n_minoActions):
                    next_s = self.__move(s, a, ma)
                    if next_s != None:
                        nextStates.append(next_s)
                prob = 1.0/len(nextStates)
                for next_s in nextStates:
                    transition_probabilities[next_s, s, a] = prob
        return transition_probabilities

    def __rewards(self, weights=None, random_rewards=None):

        rewards = np.zeros((self.n_states, self.n_actions))

        # If the rewards are not described by a weight matrix
        if weights is None:
            for s in range(self.n_states):
                for a in range(self.n_actions):
                    nextStates = []
                    for ma in range(self.n_minoActions):
                        next_s = self.__move(s, a, ma)
                        if next_s != None:
                            nextStates.append(next_s)

                    for next_s in nextStates:

                        # Rewrd for hitting a wall
                        if self.states[s][0] == self.states[next_s][0] and self.states[s][1] == self.states[next_s][1] and a != self.STAY:
                            rewards[s, a] += self.IMPOSSIBLE_REWARD
                        # reward for being eaten
                        elif self.states[next_s][0] == self.states[next_s][2] and self.states[next_s][1] == self.states[next_s][3]:
                            rewards[s, a] += self.IMPOSSIBLE_REWARD
                        # Reward for reaching the exit
                        elif self.states[s][0] == self.states[next_s][0] and self.states[s][1] == self.states[next_s][1] and self.maze[self.states[next_s][0], self.states[next_s][1]] == 2:
                            rewards[s, a] += self.GOAL_REWARD
                        # Reward for taking a step to an empty cell that is not the exit
                        else:
                            rewards[s, a] += self.STEP_REWARD
                    rewards[s, a] /= len(nextStates)

        # If the weights are descrobed by a weight matrix
        else:
            for s in range(self.n_states):
                for a in range(self.n_actions):
                    next_s = self.__move(s, a)
                    i, j = self.states[next_s]
                    # Simply put the reward as the weights o the next state.
                    rewards[s, a] = weights[i][j]

        return rewards

    def simulate(self, start, policy, method):
        if method not in methods:
            error = 'ERROR: the argument method must be in {}'.format(methods)
            raise NameError(error)

        path = list()
        if method == 'DynProg':
            # Deduce the horizon from the policy shape
            horizon = policy.shape[1]
            # Initialize current state and time
            t = 0
            s = self.map[start]
            # Add the starting position in the maze to the path
            path.append(start)
            while t < horizon-1:

                while True:
                    randomMove = np.random.randint(0, len(self.minoActions))
                    # Compute the future minotaur position given current (state, action)
                    minoRow = self.states[s][2] + \
                        self.minoActions[randomMove][0]
                    minoCol = self.states[s][3] + \
                        self.minoActions[randomMove][1]
                    try:
                        if (self.maze[minoRow, minoCol] == 1):
                            minoRow += self.minoActions[randomMove][0]
                            minoCol += self.minoActions[randomMove][1]
                    except:
                        continue

                    # Is the future position an impossible one ?
                    outside = (minoRow == -1) or (minoRow == self.maze.shape[0]) or \
                        (minoCol == -1) or (minoCol ==
                                            self.maze.shape[1]) or (self.maze[minoRow, minoCol] == 1)

                    if not outside:
                        break

                # Move to next state given the policy and the current state
                next_s = self.__move(s, policy[s, t], randomMove)
                # Add the position in the maze corresponding to the next state
                # to the path
                path.append(self.states[next_s])
                # Update time and state for next iteration
                t += 1
                s = next_s
        if method == 'ValIter':
            # Initialize current state, next state and time
            t = 1
            s = self.map[start]
            # Add the starting position in the maze to the path
            path.append(start)

            # calculate a random move for the Minotaur
            while True:
                randomMove = np.random.randint(0, len(self.minoActions))
                # Compute the future minotaur position given current (state, action)
                minoRow = self.states[s][2] + \
                    self.minoActions[randomMove][0]
                minoCol = self.states[s][3] + \
                    self.minoActions[randomMove][1]
                try:
                    if (self.maze[minoRow, minoCol] == 1):
                        minoRow += self.minoActions[randomMove][0]
                        minoCol += self.minoActions[randomMove][1]
                except:
                    continue

                # Is the future position an impossible one ?
                outside = (minoRow == -1) or (minoRow == self.maze.shape[0]) or \
                    (minoCol == -1) or (minoCol ==
                                        self.maze.shape[1]) or (self.maze[minoRow, minoCol] == 1)

                if not outside:
                    break

            # Move to next state given the policy and the current state
            next_s = self.__move(s, policy[s], randomMove)

            # Add the position in the maze corresponding to the next state
            # to the path
            path.append(self.states[next_s])
            # Loop while state is not the goal state
            while not (self.states[s][0] == self.states[next_s][0] and self.states[s][1] == self.states[next_s][1] and self.maze[self.states[next_s][0], self.states[next_s][1]] == 2):
                # Update state
                s = next_s
                while True:
                    randomMove = np.random.randint(0, len(self.minoActions))
                    # Compute the future minotaur position given current (state, action)
                    minoRow = self.states[s][2] + \
                        self.minoActions[randomMove][0]
                    minoCol = self.states[s][3] + \
                        self.minoActions[randomMove][1]
                    try:
                        if (self.maze[minoRow, minoCol] == 1):
                            minoRow += self.minoActions[randomMove][0]
                            minoCol += self.minoActions[randomMove][1]
                    except:
                        continue

                    # Is the future position an impossible one ?
                    outside = (minoRow == -1) or (minoRow == self.maze.shape[0]) or \
                        (minoCol == -1) or (minoCol ==
                                            self.maze.shape[1]) or (self.maze[minoRow, minoCol] == 1)
                    if not outside:
                        break
                # Move to next state given the policy and the current state
                next_s = self.__move(s, policy[s], randomMove)
                # Add the position in the maze corresponding to the next state
                # to the path
                path.append(self.states[next_s])
                # Update time and state for next iteration
                t += 1
        return path

    def show(self):
        print('The states are :')
        print(self.states)
        print('The actions are:')
        print(self.actions)
        print('The mapping of the states:')
        print(self.map)
        print('The rewards:')
        print(self.rewards)


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
                break

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
                             ].get_text().set_text('Player')

            grid.get_celld()[(path[i][2], path[i][3])].set_facecolor(LIGHT_RED)
            grid.get_celld()[(path[i][2], path[i][3])
                             ].get_text().set_text('Minotaur')

            playerLast = grid.get_celld()[(path[i-1][0], path[i-1][1])].xy
            playerNow = grid.get_celld()[(path[i][0], path[i][1])].xy
            deltaX = playerNow[0]-playerLast[0]
            deltaY = playerNow[1]-playerLast[1]

            plt.arrow(playerLast[0] + 0.5/rows, playerLast[1] + 0.5/cols,
                      deltaX, deltaY, width=0.005)

            minoLast = grid.get_celld()[(path[i-1][2], path[i-1][3])].xy
            minoNow = grid.get_celld()[(path[i][2], path[i][3])].xy
            deltaX = minoNow[0]-minoLast[0]
            deltaY = minoNow[1]-minoLast[1]

            plt.arrow(minoLast[0] + 0.25/rows, minoLast[1] + 0.25/cols,
                      deltaX, deltaY, width=0.005, color='red')

            if path[i][0] == path[i-1][0] and path[i][1] == path[i-1][1] and maze[(path[i][0], path[i][1])] == 2:
                print('Vi är i mål')
                grid.get_celld()[(path[i][0], path[i][1])
                                 ].set_facecolor(LIGHT_GREEN)
                grid.get_celld()[(path[i][0], path[i][1])].get_text().set_text(
                    'Player is out')
                break
        else:
            grid.get_celld()[(path[i][0], path[i][1])
                             ].set_facecolor(LIGHT_ORANGE)
            grid.get_celld()[(path[i][0], path[i][1])
                             ].get_text().set_text('Player')

            grid.get_celld()[(path[i][2], path[i][3])].set_facecolor(LIGHT_RED)
            grid.get_celld()[(path[i][2], path[i][3])
                             ].get_text().set_text('Minotaur')

        display.display(fig)
        display.clear_output(wait=True)
        # time.sleep(1)
        plt.draw()
        plt.pause(0.3)
    display.display(fig)
    display.clear_output(wait=True)
    # time.sleep(1)
    plt.show()
