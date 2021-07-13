from jmetal.core.problem import FloatProblem
from jmetal.core.solution import FloatSolution
import numpy as np
from math import sqrt, cos, sin



maze = np.load('./resources/mapImage/maparray.npy')

class Linear4(FloatProblem):

    def __init__(self, number_of_variables, x1, x2, y1, y2, lowerbound_v, lowerbound_t, upperbound_v, upperbound_t):
        super(Linear4, self).__init__()
        self.number_of_objectives = 1
        self.number_of_variables = number_of_variables
        self.number_of_constraints = int(number_of_variables / 2)
        self.obj_directions = [self.MINIMIZE]
        self.lower_bound = [lowerbound_v, lowerbound_t] * (int(number_of_variables / 2))
        self.upper_bound = [upperbound_v, upperbound_t] * (int(number_of_variables / 2))
        self.Px_departure = x1
        self.Py_departure = y1
        self.Px_arrival = x2
        self.Py_arrival = y2



    def evaluate(self, solution: FloatSolution) -> FloatSolution:

        distance = 0
        S = solution.variables
        distance_list = []

        del_t = 6
        Px_arrival = 1799 + self.Px_arrival * 10
        Py_arrival = 899 + self.Py_arrival * 10
        Px_departure = 1799 + self.Px_departure * 10
        Py_departure = 899 + self.Py_departure * 10

        x = []
        y = []
        x.append(Px_departure)
        y.append(Py_departure)
        R_diviation = 0

        for i in range(int(solution.number_of_variables / 2)-1):
            distance += S[2 * i] * del_t
            distance_list.append(S[2 * i] * del_t)
            A = x[i] + S[2 * i] * del_t * cos(S[2 * i + 1])
            B = y[i] + S[2 * i] * del_t * sin(S[2 * i + 1])

            x.append(A)
            y.append(B)


        R = sqrt((Px_arrival - x[-1]) ** 2 + (Py_arrival - y[-1]) ** 2)
        distance_list.append(R)
        x.append(Px_arrival)
        y.append(Py_arrival)



        for i in range(int(solution.number_of_variables / 2)):
            R_mean = (distance+R) / int(solution.number_of_variables / 2)

            R_diviation += sqrt((distance_list[i] - R_mean) ** 2)

        distance = distance + R + R_diviation

        solution.objectives[0] = distance
        self.evaluate_constraints(solution)

        return solution

    def evaluate_constraints(self, solution: FloatSolution) -> None:
        constraints = [0.0 for _ in range(self.number_of_constraints)]
        S = solution.variables

        del_t = 6
        x = []
        y = []
        x2 = []
        y2 = []
        Px_arrival = 1799 + self.Px_arrival * 10
        Py_arrival = 899 + self.Py_arrival * 10
        Px_departure = 1799 + self.Px_departure * 10
        Py_departure = 899 + self.Py_departure * 10
        x.append(Px_departure)
        y.append(Py_departure)
        x2.append(Px_departure)
        y2.append(Py_departure)

        for i in range(int(solution.number_of_variables / 2) - 1):
            A = x[i] + S[2 * i] * del_t * cos(S[2 * i + 1])
            B = y[i] + S[2 * i] * del_t * sin(S[2 * i + 1])

            x.append(A)
            y.append(B)
            x2.append(A)
            y2.append(B)

        x.append(Px_arrival)
        y.append(Py_arrival)
        x2.append(Px_arrival)
        y2.append(Py_arrival)

        for i in range(int(solution.number_of_variables / 2)):
            between_node_distance_x = sqrt((x[i + 1] - x[i]) ** 2)
            between_node_distance_y = sqrt((y[i + 1] - y[i]) ** 2)
            cp_x = 0
            cp_y = 0
            if x[i] > x[i + 1]:
                cp_x = x[i + 1]
            elif x[i] < x[i + 1]:
                cp_x = x[i]
            else:
                constraints[i] = -100

            if y[i] > y[i + 1]:
                cp_y = y[i + 1]
            elif y[i] < y[i + 1]:
                cp_y = y[i]
            else:
                constraints[i] = -100
            cp_x += (1 / 6) * between_node_distance_x
            cp_y += (1 / 6) * between_node_distance_y
            cp_x1 = cp_x + (1 / 6) * between_node_distance_x
            cp_y1 = cp_y + (1 / 6) * between_node_distance_y
            cp_x2 = cp_x + (1 / 6) * between_node_distance_x
            cp_y2 = cp_y + (1 / 6) * between_node_distance_y
            cp_x3 = cp_x + (1 / 6) * between_node_distance_x
            cp_y3 = cp_y + (1 / 6) * between_node_distance_y
            cp_x4 = cp_x + (1 / 6) * between_node_distance_x
            cp_y4 = cp_y + (1 / 6) * between_node_distance_y

            if maze[int(1799 - y[i + 1])][int(x[i + 1])] == 1 or maze[int(1799 - y[i + 1] + 1)][
                int(x[i + 1]) + 1] == 1 or maze[int(1799 - y[i + 1] - 1)][int(x[i + 1]) - 1] == 1 \
                    or maze[int(1799 - cp_y) + 1][int(cp_x) + 1] == 1 or maze[int(1799 - cp_y) - 1][int(cp_x) - 1] == 1 \
                    or maze[int(1799 - cp_y1) + 1][int(cp_x1) + 1] == 1 or maze[int(1799 - cp_y1) - 1][
                int(cp_x1) - 1] == 1 \
                    or maze[int(1799 - cp_y2) + 1][int(cp_x2) + 1] == 1 or maze[int(1799 - cp_y2) - 1][
                int(cp_x2) - 1] == 1 \
                    or maze[int(1799 - cp_y3) + 1][int(cp_x3) + 1] == 1 or maze[int(1799 - cp_y3) - 1][
                int(cp_x3) - 1] == 1 \
                    or maze[int(1799 - cp_y4) + 1][int(cp_x4) + 1] == 1 or maze[int(1799 - cp_y4) - 1][
                int(cp_x4) - 1] == 1 \
                    or maze[int(1799 - cp_y)][int(cp_x)] == 1 or maze[int(1799 - cp_y)][int(cp_x)] == 1 \
                    or maze[int(1799 - cp_y1)][int(cp_x1)] == 1 or maze[int(1799 - cp_y1)][int(cp_x1)] == 1 \
                    or maze[int(1799 - cp_y2)][int(cp_x2)] == 1 or maze[int(1799 - cp_y2)][int(cp_x2)] == 1 \
                    or maze[int(1799 - cp_y3)][int(cp_x3)] == 1 or maze[int(1799 - cp_y3)][int(cp_x3)] == 1 \
                    or maze[int(1799 - cp_y4)][int(cp_x4)] == 1 or maze[int(1799 - cp_y4)][int(cp_x4)] == 1:
                constraints[i] = -100

        solution.constraints = constraints

    def get_name(self) -> str:
        return 'Linear4'