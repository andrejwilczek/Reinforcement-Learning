import numpy as np
import problem_2_maze as mz
import matplotlib.pyplot as plt

# Andrej Wilzcek 880707-7477
# Kildo Alias 971106-7430


maze = np.array([
    [2, 0, 0, 0, 0, 2],
    [0, 0, 1, 0, 0, 0],
    [2, 0, 0, 0, 0, 2]
])
# # with the convention
# # 0 = empty cell
# # 1 = police station
# # 2 = bank

start = (0, 0, 1, 2)
env = mz.Maze(maze)
method = 'ValIter'
epsilon = 0.01
gamma = 0.1

gammavec = [i/100.0 for i in range(1, 100, 1)]
Vvec = []
# mz.draw_maze(maze)
# env.show()
# for gamma in gammavec:
#     print('gamma: ', gamma)
#     V, policy = mz.value_iteration(env, gamma, epsilon)
#     Vvec.append(V[8])

# plt.plot(gammavec, Vvec)
# plt.xlabel('Lambda')
# plt.ylabel('Value function')
# plt.title('Value function evaluated in inital state')
# plt.grid()
# plt.show()


V, policy = mz.value_iteration(env, gamma, epsilon)
# print(policy)
path = env.simulate(start, policy, method)
mz.animate_solution(maze, path)
