import numpy as np
import problem_1_maze as mz
import matplotlib.pyplot as plt

# Andrej Wilzcek 880707-7477
# Kildo Alias 971106-7430


def task_b(iterations):

    iterations = iterations
    method = 'DynProg'
    start = (0, 0, 6, 5)
    horizonvec = [i for i in range(26)]
    stayvec = [False, True]

    for j in stayvec:

        env = mz.Maze(maze, minoStay=j)
        outvec = []

        # Finite horizon
        for horizon in horizonvec:
            print(horizon)

            # horizon = 20
            # Solve the MDP problem with dynamic programming
            V, policy = mz.dynamic_programming(env, horizon)
            goals = 0
            for i in range(iterations):
                # # Simulate the shortest path starting from position A
                path = env.simulate(start, policy, method)
                counter = 0
                for step in path:
                    if step[0] == start[2] and step[1] == start[3]:
                        counter += 1
                    else:
                        counter = 0

                    if counter == 2:
                        goals += 1
                        break
            outvec.append(goals/iterations)

        plt.plot(horizonvec, outvec)
    plt.xlabel('T')
    plt.ylabel('Exit probability')
    plt.title('Probability of exiting the maze')
    plt.legend(['Minotaur can\'t stay', 'Minotaur can stay'])
    plt.show()

# # Show the shortest path
# mz.animate_solution(maze, path)


def task_c(iterations):

    env = mz.Maze(maze, minoStay=False)
    start = (0, 0, 6, 5)
    method = 'ValIter'
    gamma = 1-(1/30)
    epsilon = 0.01
    print('Calculating policy...')
    V, policy = mz.value_iteration(env, gamma, epsilon)

    iterations = iterations

    goals = 0
    print('Simulating...')
    for i in range(iterations):
        if i % (iterations/10) == 0:
            print(i, ' iterations')
        # Simulate the shortest path starting from position A
        path = env.simulate(start, policy, method)
        counter = 0
        for step in path:
            if step[0] == start[2] and step[1] == start[3]:
                counter += 1
            else:
                counter = 0

            if step[0] == step[2] and step[1] == step[3]:
                print('You have been munched.. :(')
                break

            if counter == 2:
                goals += 1
                break
    outprob = (goals/iterations)
    print(outprob, ' probability of survival')


def animate_Valiter(policy, start, method, maze):
    path = env.simulate(start, policy, method)
    mz.animate_solution(maze, path)


if __name__ == "__main__":

    # Description of the maze as a numpy array
    maze = np.array([
        [0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0, 1, 1, 1],
        [0, 0, 1, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 1, 1, 0],
        [0, 0, 0, 0, 1, 2, 0, 0]
    ])
    # with the convention
    # 0 = empty cell
    # 1 = obstacle
    # 2 = exit of the Maze

    task_b(10)

    task_c(10000)

    env = mz.Maze(maze, minoStay=False)
    start = (0, 0, 6, 5)
    method = 'ValIter'
    gamma = 1-(1/30)
    epsilon = 0.01
    print('Calculating policy...')
    V, policy = mz.value_iteration(env, gamma, epsilon)
    for i in range(10):
        animate_Valiter(policy, start, method, maze)
