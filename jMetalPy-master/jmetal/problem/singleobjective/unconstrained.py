from math import sqrt, cos, sin, atan, acos, pi
import random
import numpy as np
from jmetal.core.problem import BinaryProblem, FloatProblem
from jmetal.core.solution import BinarySolution, FloatSolution

"""
.. module:: unconstrained
   :platform: Unix, Windows
   :synopsis: Unconstrained test problems for single-objective optimization

.. moduleauthor:: Antonio J. Nebro <antonio@lcc.uma.es>, Antonio Benítez-Hidalgo <antonio.b@uma.es>
"""


class OneMax(BinaryProblem):

    def __init__(self, number_of_bits: int = 256):
        super(OneMax, self).__init__()
        self.number_of_bits = number_of_bits
        self.number_of_objectives = 1
        self.number_of_variables = 1
        self.number_of_constraints = 0

        self.obj_directions = [self.MINIMIZE]
        self.obj_labels = ['Ones']

    def evaluate(self, solution: BinarySolution) -> BinarySolution:
        counter_of_ones = 0
        for bits in solution.variables[0]:
            if bits:
                counter_of_ones += 1

        solution.objectives[0] = -1.0 * counter_of_ones

        return solution

    def create_solution(self) -> BinarySolution:
        new_solution = BinarySolution(number_of_variables=1, number_of_objectives=1)
        new_solution.variables[0] = \
            [True if random.randint(0, 1) == 0 else False for _ in range(self.number_of_bits)]
        return new_solution

    def get_name(self) -> str:
        return 'OneMax'




class Sphere(FloatProblem):

    def __init__(self, number_of_variables: int = 1):
        super(Sphere, self).__init__()
        self.number_of_objectives = 1
        self.number_of_variables = number_of_variables
        self.number_of_constraints = 0

        self.obj_directions = [self.MINIMIZE]
        self.obj_labels = ['f(x)']

        self.lower_bound = [1 for _ in range(number_of_variables)]
        self.upper_bound = [10 for _ in range(number_of_variables)]

    def evaluate(self, solution: FloatSolution) -> FloatSolution:

        r = solution.variables

        P_list = []



        P = -(0.00005*((r[0])**6) - 0.0016*((r[0])**5) + 0.0178*((r[0])**4) - 0.0869*((r[0])**3) + 0.1268*((r[0])**2) + 0.2264*(r[0]) + 0.0007)
        P_list.append(P)


        P_Min = min(P_list)
        print(P_Min)

        solution.objectives[0] = P_Min

        return solution

    def get_name(self) -> str:
        return 'Sphere'


class Linear(FloatProblem):
    def __init__(self, number_of_variables: int=9):
        super(Linear, self).__init__()
        self.number_of_objectives = 1
        self.number_of_variables = number_of_variables
        self.number_of_constraints = 8
        self.obj_directions = [self.MINIMIZE]
        self.lower_bound = [0 for _ in range(number_of_variables)]
        self.upper_bound = [5.12 for _ in range(number_of_variables)]
        self.lower_bound[6:] = [0 for _ in range(number_of_variables)]
        self.upper_bound[6:] = [45*pi/180 for _ in range(number_of_variables)]

    def evaluate(self, solution: FloatSolution) -> FloatSolution:
         distance = 0
         S = solution.variables





         for i in range(0,9):
             distance += sqrt((S[2*i+2] - S[2*i])**2+(S[2*i+3]-S[2*i+1])**2)

         solution.objectives[0] = distance
         self.evaluate_constraints(solution)

         return solution

    def evaluate_constraints(self, solution: FloatSolution) -> None:
        constraints = [0.0 for _ in range(self.number_of_constraints)]
        S = solution.variables
        e = 0.01
        constraints[0] = -S[0]-e
        constraints[1] = S[0]-e
        constraints[2] = -S[1]-e
        constraints[3] = S[1]-e
        #constraints[4] = 2.5-e-S[10]
        #constraints[5] = -2.5-e+S[10]
        #constraints[6] = 2.5-e-S[11]
        #constraints[7] = -2.5-e+S[11]
        constraints[4] = 5-e-S[18]
        constraints[5] = -5-e+S[18]
        constraints[6] = 5-e-S[19]
        constraints[7] = -5-e+S[19]
        solution.constraints = constraints

    def get_name(self) -> str:
        return 'Linear'

class Linear2(FloatProblem):
    def __init__(self, number_of_variables: int=10):
        super(Linear2, self).__init__()
        self.number_of_objectives = 1
        self.number_of_variables = number_of_variables
        self.number_of_constraints = 8
        self.obj_directions = [self.MINIMIZE]
        self.lower_bound = [0 for _ in range(number_of_variables)]
        self.upper_bound = [6 for _ in range(number_of_variables)]

    def evaluate(self, solution: FloatSolution) -> FloatSolution:
         distance = 0
         S = solution.variables
         velocity_list = []
         theta_list = []

         for i in range(0,4):

             distance += (sqrt((S[2*i+2] - S[2*i])**2))/(cos(atan((sqrt((S[2*i+3]-S[2*i+1])**2))/
                                                                                (sqrt((S[2*i+2] - S[2*i])**2)))))


             #(S[2*i+2] - S[2*i])**2)
             delta_t = 0.5
             velocity = distance/delta_t
             velocity_list.append(velocity)
             theta = atan((sqrt((S[2*i+3]-S[2*i+1])**2))/ (sqrt((S[2*i+2] - S[2*i])**2)))
             theta_list.append(theta)
         print(velocity_list)
         print(theta_list)


         solution.objectives[0] = distance
         self.evaluate_constraints(solution)

         return solution

    def evaluate_constraints(self, solution: FloatSolution) -> None:
        constraints = [0.0 for _ in range(self.number_of_constraints)] # 제약 조건 갯수를 가져옴
        S = solution.variables
        e = 0.01
        constraints[0] = -S[0] - e
        constraints[1] = S[0] - e
        constraints[2] = -S[1] - e
        constraints[3] = S[1] - e
        constraints[4] = 5 - e - S[8]
        constraints[5] = -5 - e + S[8]
        constraints[6] = 5 - e - S[9]
        constraints[7] = -5 - e + S[9]

        solution.constraints = constraints

    def get_name(self) -> str:
        return 'Linear2'



class Linear4(FloatProblem):

    def __init__(self, number_of_variables: int = 42):
        super(Linear4, self).__init__()
        self.number_of_objectives = 1
        self.number_of_variables = number_of_variables
        self.number_of_constraints = int(number_of_variables/2 )
        self.obj_directions = [self.MINIMIZE]
        self.lower_bound = [0, -pi/2, 0, -pi/2, 0, -pi/2, 0, -pi/2, 0, -pi/2, 0, -pi/2, 0, -pi/2, 0, -pi/2, 0, -pi/2, 0, -pi/2,
                            0, -pi/2, 0, -pi/2, 0, -pi/2, 0, -pi/2, 0, -pi/2, 0, -pi/2, 0, -pi/2, 0, -pi/2, 0, -pi/2, 0, -pi/2, 0, -pi/2]
        self.upper_bound = [10000, -pi/2, 10000, pi/2, 10000, pi/2, 10000, pi/2, 10000, pi/2, 10000, pi/2, 10000, pi/2, 10000, pi/2, 10000, pi/2, 10000, pi/2,
                            10000, pi/2, 10000, pi/2, 10000, pi/2, 10000, pi/2, 10000, pi/2, 10000, pi/2, 10000, pi/2, 10000, pi/2, 10000, pi/2, 10000, pi/2, 10000, pi/2]

    def evaluate(self, solution: FloatSolution) -> FloatSolution:


        distance = 0
        S = solution.variables
        del_t = 1
        Px_arrival = 30
        Py_arrival = 0
        Px_departure = 0
        Py_departure = 0
        x = []
        y = []
        x.append(Px_departure)
        y.append(Py_departure)
        R_diviation = 0
        S[0] = 1.7
        S[1] = 0.8

        for i in range(int(solution.number_of_variables/2)-1):
            distance += S[2 * i] * del_t
            A =x[i] + S[2 * i] * del_t * cos(S[2 * i + 1])
            B =y[i] + S[2 * i] * del_t * sin(S[2 * i + 1])
            x.append(A)
            y.append(B)

        R = sqrt((Px_arrival-x[-1])**2 + (Py_arrival-y[-1])**2)

        #print(R)

        S[41] = acos(sqrt((Px_arrival - x[-1])**2) / R)

        xi = x[-1] + R * cos(S[-1])
        yi = x[-1] + R * sin(S[-1])
        x.append(xi)
        y.append(yi)





        for i in range(int(solution.number_of_variables/2)):
            R_mean = (distance + R) / int(solution.number_of_variables/2)

            R_diviation += sqrt((S[2*i]*del_t - R_mean)**2)






        distance = distance + R + R_diviation
        #print(distance)



        solution.objectives[0] = distance
        self.evaluate_constraints(solution)
        #for i in range(len(x)):
            #print(x[i], y[i])


        return solution

    def evaluate_constraints(self, solution: FloatSolution) -> None:
        constraints = [0.0 for _ in range(self.number_of_constraints)]
        S = solution.variables
        del_t = 1
        x = []
        y = []
        x.append(0)
        y.append(0)
        x_Obstacle1 = 15
        y_Obstacle1 = 5
        #x_Obstacle2 = 18
        #y_Obstacle2 = 12
        #x_Obstacle3 = 20
        #y_Obstacle3 = 25

        for i in range(int(solution.number_of_variables/2)-1):
            A = x[i] + S[2 * i] * del_t * cos(S[2 * i + 1])
            B = y[i] + S[2 * i] * del_t * sin(S[2 * i + 1])
            x.append(A)
            y.append(B)

        e = 1
        for i in range(int(((solution.number_of_variables/2)-1))):
            constraints[i] = sqrt((x_Obstacle1-x[i+1])**2 + (y_Obstacle1-y[i+1])**2) - 5 - e
            #constraints[i+20] = sqrt((x_Obstacle2-x[i+1])**2 + (y_Obstacle2-y[i+1])**2) - 4 - e
            #constraints[i+40] = sqrt((x_Obstacle3-x[i+1])**2 + (y_Obstacle3-y[i+1])**2) - 4 - e
            constraints[20] = y[i] - 0.2



        solution.constraints = constraints

    def get_name(self) -> str:
        return 'Linear4'


class Rastrigin(FloatProblem):

    def __init__(self, number_of_variables: int = 10):
        super(Rastrigin, self).__init__()
        self.number_of_objectives = 1
        self.number_of_variables = number_of_variables
        self.number_of_constraints = 0

        self.obj_directions = [self.MINIMIZE]
        self.obj_labels = ['f(x)']

        self.lower_bound = [-5.12 for _ in range(number_of_variables)]
        self.upper_bound = [5.12 for _ in range(number_of_variables)]

        self.lower_bound = self.number_of_variables * [-5.0]
        self.upper_bound = self.number_of_variables * [5.0]

    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        a = 10.0
        result = a * solution.number_of_variables
        x = solution.variables

        for i in range(solution.number_of_variables):
            result += x[i] * x[i] - a * cos(2 * pi * x[i])

        solution.objectives[0] = result

        return solution

    def get_name(self) -> str:
        return 'Rastrigin'


class SubsetSum(BinaryProblem):

    def __init__(self, C: int, W: list):
        """ The goal is to find a subset S of W whose elements sum is closest to (without exceeding) C.

        :param C: Large integer.
        :param W: Set of non-negative integers."""
        super(SubsetSum, self).__init__()
        self.C = C
        self.W = W

        self.number_of_bits = len(self.W)
        self.number_of_objectives = 1
        self.number_of_variables = 1
        self.number_of_constraints = 0

        self.obj_directions = [self.MAXIMIZE]
        self.obj_labels = ['Sum']

    def evaluate(self, solution: BinarySolution) -> BinarySolution:
        total_sum = 0.0

        for index, bits in enumerate(solution.variables[0]):
            if bits:
                total_sum += self.W[index]

        if total_sum > self.C:
            total_sum = self.C - total_sum * 0.1

            if total_sum < 0.0:
                total_sum = 0.0

        solution.objectives[0] = -1.0 * total_sum

        return solution

    def create_solution(self) -> BinarySolution:
        new_solution = BinarySolution(number_of_variables=self.number_of_variables,
                                      number_of_objectives=self.number_of_objectives)
        new_solution.variables[0] = \
            [True if random.randint(0, 1) == 0 else False for _ in range(self.number_of_bits)]

        return new_solution

    def get_name(self) -> str:
        return 'Subset Sum'
