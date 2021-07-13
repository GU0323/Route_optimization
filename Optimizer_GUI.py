from PyQt5.QtWebEngineWidgets import QWebEngineView
import sys
from PyQt5.QtWidgets import *
from PyQt5 import uic
from jmetal.algorithm.multiobjective.nsgaii import NSGAII
from jmetal.operator import SBXCrossover, PolynomialMutation
from jmetal.util.comparator import DominanceComparator
from jmetal.util.solution import print_function_values_to_file, print_variables_to_file
from jmetal.util.termination_criterion import StoppingByEvaluations
import io
import folium
from math import cos, sin
from defined_problem import Linear4




form_ui = uic.loadUiType("./GUI/optimizer_route.ui")[0]

class MyWindow(QMainWindow, form_ui):

    def __init__(self):
        super(MyWindow, self).__init__()
        self.setupUi(self)
        self.start_button.clicked.connect(self.OPtimizer)
        self.loadmap.clicked.connect(self.LoadMap)
        self.applybutton.clicked.connect(self.Apply)
        self.Node = 0
        self.departure_lat = 0
        self.departure_lon = 0
        self.arrival_lat = 0
        self.arrival_lon = 0
        self.text = ''
        self.text2 = ''

        self.x2 = []
        self.y2 = []
        self.webView = QWebEngineView()
        self.m = folium.Map(
            location=[0, 0],
            tiles='Stamen Terrain',
            zoom_start=2
        )

        data = io.BytesIO()
        self.m.save(data, close_file=False)
        self.webView.setHtml(data.getvalue().decode())
        self.verticalLayout.addWidget(self.webView)

        self.departure.activated[str].connect(self.onActedate_departure)
        self.arrival.activated[str].connect(self.onActedate_arrival)



    def Apply(self):
        lat = (float(self.departure_lat) + float(self.arrival_lat)) / 2
        lon = (float(self.departure_lon) + float(self.arrival_lon)) / 2
        self.m = folium.Map(
            location=[lat , lon],
            tiles='Stamen Terrain',
            zoom_start=4
        )
        data = io.BytesIO()
        self.m.save(data, close_file=False)

        self.webView.setHtml(data.getvalue().decode())

    def onActedate_departure(self, text):
        self.text = text
        if self.text == 'Busan':
            self.departure_lat = 35.046
            self.departure_lon = 129.627
        elif self.text == 'HociMinh':
            self.departure_lat = 9.62
            self.departure_lon = 107.992
        return

    def onActedate_arrival(self, text):
        self.text2 = text
        if self.text2 == 'Busan':
            self.arrival_lat = 35.046
            self.arrival_lon = 129.627
        elif self.text2 == 'HochiMinh':
            self.arrival_lat = 9.62
            self.arrival_lon = 107.992
        return
    def OPtimizer(self):
        self.de = []
        self.ar = []
        self.Node = int(self.Node_line.text())

        self.lb = []
        self.ub = []
        lower_bound = str(self.lowerbound.text())
        upper_bound = str(self.upperbound.text())
        self.lb = lower_bound.split(',')
        self.ub = upper_bound.split(',')
        self.lower_bound_v = float(self.lb[0])
        self.lower_bound_t = float(self.lb[1])
        self.upper_bound_v = float(self.ub[0])
        self.upper_bound_t = float(self.ub[1])
        self.max_evaluations = int(self.maxevaluations.text())
        self.population_size = int(self.populationsize.text())
        self.lines = []
        self.point = ()

        problem = Linear4(self.Node, self.departure_lon, self.arrival_lon, self.departure_lat, self.arrival_lat, self.lower_bound_v, self.lower_bound_t, self.upper_bound_v, self.upper_bound_t)

        max_evaluations = self.max_evaluations
        algorithm = NSGAII(
            problem=problem,
            population_size=self.population_size,
            offspring_population_size=self.population_size,
            mutation=PolynomialMutation(probability=1.0 / problem.number_of_variables, distribution_index=20.0),
            crossover=SBXCrossover(probability=0.9, distribution_index=20.0),
            termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations),
            dominance_comparator=DominanceComparator()
        )

        algorithm.run()
        front = algorithm.get_result()

        # Save results to file
        print_function_values_to_file(front, 'FUN3_부산_호치민(node=40).' + algorithm.get_name() + "-" + problem.get_name())
        print_variables_to_file(front, 'VAR3_linear_부산_호치민(node=40).' + algorithm.get_name() + "-" + problem.get_name())

        file_open = open('VAR3_linear_부산_호치민(node=40).' + algorithm.get_name() + "-" + problem.get_name(), 'r',
                         encoding='utf-8')

        file_open2 = open('FUN3_부산_호치민(node=40).' + algorithm.get_name() + "-" + problem.get_name(), 'r',
                          encoding='utf-8')
        obj_line = file_open2.readlines()
        obj = obj_line[0]
        print(obj)
        file_open2.close()

        line = file_open.readlines()
        word = line[0].split(" ")
        x = []
        y = []
        Px_arrival = 1799 + self.arrival_lon * 10
        Py_arrival = 899 + self.arrival_lat * 10
        Px_departure = 1799 + self.departure_lon * 10
        Py_departure = 899 + self.departure_lat * 10

        x.append(Px_departure)
        y.append(Py_departure)
        del_t = 6

        for i in range(int(self.Node / 2)-1):
            xi = x[i] + float(word[2 * i]) * del_t * cos(float(word[2 * i + 1]))
            yi = y[i] + float(word[2 * i]) * del_t * sin(float(word[2 * i + 1]))

            x.append(xi)
            y.append(yi)

        x.append(Px_arrival)
        y.append(Py_arrival)

        for i in range(len(x)):
            print(x[i], y[i])

        for i in range(len(x)):
            py = (y[i]-899) / 10
            px = (x[i]-1799) / 10

            self.x2.append(px)
            self.y2.append(py)

            print(self.x2[i], self.y2[i])

        for i in range(len(self.x2)):
            li = []
            li.append(self.y2[i])
            li.append(self.x2[i])
            self.lines.append(li)
        folium.PolyLine(
            locations=self.lines,
            color='red',
            tooltip='path'
        ).add_to(self.m)
        for i, j in zip(self.x2, self.y2):
            folium.Circle(
                location=(j, i),
                radius=10000,
                color='yellow'
            ).add_to(self.m)

        folium.Marker(
            [self.y2[0], self.x2[0]],
            popup='<b>waypoint</b>',
            icon=folium.Icon(color='green', icon='bookmark', tooltip='Departure')

        ).add_to(self.m)
        folium.Marker(
            [self.y2[-1], self.x2[-1]],
            popup='<b>waypoint</b>',
            icon=folium.Icon(color='green', icon='bookmark', tooltip='Arrival')
        ).add_to(self.m)

        data = io.BytesIO()
        self.m.save(data, close_file=False)
        self.webView.setHtml(data.getvalue().decode())


        self.label_distance.clear()
        self.label_distance.setText('{0}'.format(obj))


        print('Algorithm (continuous problem): ' + algorithm.get_name())
        print('Problem: ' + problem.get_name())
        print('Computing time: ' + str(algorithm.total_computing_time))
        file_open.close()

    def LoadMap(self):

        self.m = folium.Map(
            location=[0, 0],
            tiles='Stamen Terrain',
            zoom_start=2
        )

        data = io.BytesIO()
        self.m.save(data, close_file=False)
        self.webView.setHtml(data.getvalue().decode())




if __name__ == "__main__":
    app = QApplication(sys.argv)
    myWindow = MyWindow()
    myWindow.show()
    app.exec_()






