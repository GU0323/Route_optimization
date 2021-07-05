import sys
from PyQt5.QtWidgets import *
from PyQt5 import uic
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PyQt5.QtGui import *
from PyQt5.QtCore import Qt
from jmetal.algorithm.multiobjective.nsgaii import NSGAII
from jmetal.operator import SBXCrossover, PolynomialMutation
from jmetal.util.comparator import DominanceComparator
from jmetal.util.solution import print_function_values_to_file, print_variables_to_file
from jmetal.util.termination_criterion import StoppingByEvaluations
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas



from math import sqrt, cos, sin, atan, acos, pi

from jmetal.core.problem import BinaryProblem, FloatProblem
from jmetal.core.solution import BinarySolution, FloatSolution

form_ui = uic.loadUiType("optimizer.ui")[0]





class Linear4(FloatProblem):


    def __init__(self, number_of_variables, x1, x2, y1, y2, Constrained_Number):

        super(Linear4, self).__init__()
        self.number_of_objectives = 1
        self.number_of_variables = number_of_variables
        self.number_of_constraints = Constrained_Number
        self.obj_directions = [self.MINIMIZE]
        #self.bound = [10, pi/2] * (int(number_of_variables/2))
        self.lower_bound = [0 for _ in range(number_of_variables)]
        #self.lower_bound = [0, -pi] * (int(number_of_variables/2))
        self.upper_bound = [1000, pi] * (int(number_of_variables/2))
        #self.lower_bound = [4.5, 0.473619, 4.5, 0.45367, 4.5, 0.405556, 4.5, -0.10795, 4.5, -0.64639, 4.5, -0.64602, 4.5, 1.462392]
        #self.upper_bound = [5, 0.673619, 5, 0.65367, 5, 0.605556, 5, 0.092052, 5, -0.44639, 5, -0.44602, 5, 1.662392]
        print(self.upper_bound)


        self.Px_departure = x1
        self.Py_departure = y1
        self.Px_arrival = x2
        self.Py_arrival = y2


    def evaluate(self, solution: FloatSolution) -> FloatSolution:




        distance = 0
        S = solution.variables

        del_t = 1
        Px_arrival = self.Px_arrival
        Py_arrival = self.Py_arrival
        Px_departure = self.Px_departure
        Py_departure = self.Py_departure
        x = []
        y = []
        x.append(Px_departure)
        y.append(Py_departure)
        R_diviation = 0




        for i in range(int(solution.number_of_variables/2)-1):
            distance += S[2 * i] * del_t
            A =x[i] + S[2 * i] * del_t * cos(S[2 * i + 1])
            B =y[i] + S[2 * i] * del_t * sin(S[2 * i + 1])
            x.append(A)
            y.append(B)

        R = sqrt((Px_arrival-x[-1])**2 + (Py_arrival-y[-1])**2)



        x.append(Px_arrival)
        y.append(Py_arrival)





        for i in range(int(solution.number_of_variables/2)):
            R_mean = (distance + R) / int(solution.number_of_variables/2)

            R_diviation += sqrt((S[2*i]*del_t - R_mean)**2)




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
        Px_departure = self.Px_departure
        Py_departure = self.Py_departure

        x.append(Px_departure)
        y.append(Py_departure)

        x_Obstacle1 = 6
        y_Obstacle1 = 7
        x_Obstacle2 = 14
        y_Obstacle2 = 16
        x_Obstacle3 = 23
        y_Obstacle3 = 20

        Px_arrival = self.Px_arrival
        Py_arrival = self.Py_arrival

        for i in range(int(solution.number_of_variables/2)-1):
            A = x[i] + S[2 * i] * del_t * cos(S[2 * i + 1])
            B = y[i] + S[2 * i] * del_t * sin(S[2 * i + 1])
            x.append(A)
            y.append(B)

        x.append(Px_arrival)
        y.append(Py_arrival)

        e = 1
        for i in range(int(solution.number_of_variables/2)):
            constraints[i] = sqrt((x_Obstacle1-x[i+1])**2 + (y_Obstacle1-y[i+1])**2) - 3 -e
            constraints[i+10] = sqrt((x_Obstacle2-x[i+1])**2 + (y_Obstacle2-y[i+1])**2) - 4 -e
            constraints[i+20] = sqrt((x_Obstacle3 - x[i+1]) ** 2 + (y_Obstacle3 - y[i+1]) ** 2) - 3 -e
            #constraints[i+27] = sqrt((x_Obstacle4 - x[i+1]) ** 2 + (y_Obstacle4 - y[i + 1]) ** 2) - 15 - e


        '''
        for i in range(0, 8):
            constraints[i] = sqrt((x_Obstacle1 - x[i+1]) ** 2 + (y_Obstacle1 - y[i+1]) ** 2) - 3 - e
        for i in range(8, 20):
            constraints[i] = sqrt((x_Obstacle2 - x[i + 1]) ** 2 + (y_Obstacle2 - y[i + 1]) ** 2) - 4 - e

        for i in range(15, 20):
            constraints[i+5] = sqrt((x_Obstacle3 - x[i+1]) ** 2 + (y_Obstacle3 - y[i+1]) ** 2) - 4 - e

        for i in range(11, 20):
            constraints[i+14] = sqrt((x_Obstacle4 - x[i+1]) ** 2 + (y_Obstacle4 - y[i+1]) ** 2) - 15 - e

        for i in range(11, 20):
            constraints[i + 14] = sqrt((x_Obstacle5 - x[i + 1]) ** 2 + (y_Obstacle5 - y[i + 1]) ** 2) - 15 - e
        '''


        solution.constraints = constraints

    def get_name(self) -> str:
        return 'Linear4'


class MyWindow(QMainWindow, form_ui):

    def __init__(self):
        super(MyWindow, self).__init__()
        self.setupUi(self)
        self.fig = plt.Figure()
        self.canvas = FigureCanvas(self.fig)
        self.graph_verticalLayout.addWidget(self.canvas)
        self.Startbutton.clicked.connect(self.OPtimizer)
        self.Px_departure = 0
        self.Py_departure = 0
        self.Px_arrival = 0
        self.Py_arrival = 0
        self.Node = 0
        self.de = []
        self.ar = []
        self.Constrained_Number = 0
        self.button2.clicked.connect(self.ResetButton)


    def OPtimizer(self):
        self.de = []
        self.ar = []
        departure = str(self.lineEdit_2.text())
        arrival = str(self.lineEdit_3.text())
        self.Constrained_Number = int(self.lineEdit_4.text())
        self.de = departure.split(',')
        self.ar = arrival.split(',')
        self.Px_departure = float(self.de[0])
        self.Py_departure = float(self.de[1])
        self.Px_arrival = float(self.ar[0])
        self.Py_arrival = float(self.ar[1])
        self.Node = int(self.lineEdit.text())






        problem = Linear4(self.Node, self.Px_departure, self.Px_arrival, self.Py_departure, self.Py_arrival, self.Constrained_Number)

        max_evaluations = 20000
        algorithm = NSGAII(
            problem=problem,
            population_size=100,
            offspring_population_size=100,
            mutation=PolynomialMutation(probability=1.0 / problem.number_of_variables, distribution_index=20.0),
            crossover=SBXCrossover(probability=0.9, distribution_index=20.0),
            termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations),
            dominance_comparator=DominanceComparator()
        )

        algorithm.run()
        front = algorithm.get_result()

        # Save results to file
        print_function_values_to_file(front, 'FUN3_linear_초기조건입력완료2.' + algorithm.get_name() + "-" + problem.get_name())
        print_variables_to_file(front, 'VAR3_linear_초기조건입력완료2.' + algorithm.get_name() + "-" + problem.get_name())

        file_open = open('VAR3_linear_초기조건입력완료2.' + algorithm.get_name() + "-" + problem.get_name(), 'r', encoding='utf-8')
        line = file_open.readlines()
        word = line[-1].split(" ")
        x = []
        y = []
        y2 = []
        Px_arrival = self.Px_arrival
        Py_arrival = self.Py_arrival
        Px_departure = self.Px_departure
        Py_departure = self.Py_departure
        x.append(Px_departure)
        y.append(Py_departure)
        del_t = 1
        for i in range(int(self.Node/2-1)):
            xi = x[i] + float(word[2 * i]) * del_t * cos(float(word[2 * i + 1]))
            yi = y[i] + float(word[2 * i]) * del_t * sin(float(word[2 * i + 1]))
            x.append(xi)
            y.append(yi)
            y2.append(yi)

        x.append(Px_arrival)
        y.append(Py_arrival)
        y2.append(Py_arrival)
        for i in range(len(x)):
            print(x[i], y[i])

        self.fig.clear()
        ax = self.fig.add_subplot(111)
        circle1 = patches.Circle((6, 7), 3, color="r")
        circle2 = patches.Circle((14, 16), 4, color="r")
        circle3 = patches.Circle((23, 20), 3, color="r")
        ax.add_artist(circle1)
        ax.add_artist(circle2)
        ax.add_artist(circle3)

        ax.plot(x, y)
        ax.set_xlim([0, 30])
        ax.set_ylim([0, 30])
        ax.set_xlabel("x")
        ax.set_xlabel("y")
        ax.set_title("Optimizer_route")

        ax.grid()
        self.canvas.draw()

        file_open.close()

        print('Algorithm (continuous problem): ' + algorithm.get_name())
        print('Problem: ' + problem.get_name())
        print('Computing time: ' + str(algorithm.total_computing_time))



        return

    def ResetButton(self):
        #self.fig.clear()
        self.lineEdit.clear()
        self.lineEdit_2.clear()
        self.lineEdit_3.clear()
        self.lineEdit_4.clear()




if __name__=="__main__":
    app = QApplication(sys.argv)
    myWindow = MyWindow()
    myWindow.show()
    app.exec_()