from PyQt5.QtWebEngineWidgets import QWebEngineView
import sys
from PyQt5.QtWidgets import *
from PyQt5 import uic
from PyQt5 import QtTest
import io
import folium
from functions.Optimizer_run import ThreadClass
from functions.Optimizer_run import ThreadClass2
from folium.features import CustomIcon
import numpy as np
from PyQt5.QtGui import QIcon
from functions.jin_buk_theta import jinbuk
from vincenty import vincenty
from datetime import datetime

Wave_array = np.load('./resources/mapImage/Wave_Heigt_gird.npy')
form_ui = uic.loadUiType("./GUI/optimizer_route.ui")[0]
Node = 0
lower_bound_v = 0
lower_bound_t = 0
upper_bound_v = 0
upper_bound_t = 0
max_evaluations = 0
population_size = 0
departure_lon = 0
departure_lat = 0
arrival_lon = 0
arrival_lat = 0
del_t = 0
draught = 0

class MyWindow(QMainWindow, form_ui):

    def __init__(self):
        super(MyWindow, self).__init__()
        self.setupUi(self)
        self.setWindowTitle('optimizer_route')
        self.setWindowIcon(QIcon('./GUI/ship_wheel_icon.png'))
        self.threadclass = ThreadClass()
        self.threadclass2 = ThreadClass2()
        self.start_button.clicked.connect(self.OPtimizer)
        self.applybutton.clicked.connect(self.Apply)
        self.loadweather_button.clicked.connect(self.loadweather)
        self.departure.activated[str].connect(self.onActedate_departure)
        self.arrival.activated[str].connect(self.onActedate_arrival)
        self.astar_button.clicked.connect(self.astar)
        self.text = ''
        self.text2 = ''
        self.lb = []
        self.ub = []
        self.lower_bound = ''
        self.upper_bound = ''
        self.lower_bound_v = 0
        self.lower_bound_t = 0
        self.upper_bound_v = 0
        self.upper_bound_t = 0
        self.max_evaluations = 0
        self.population_size = 0
        self.Node = 0
        self.del_t = 0
        self.draught = 0
        self.D_dateVar = None
        self.A_dateVar = None
        self.D_timeVar = None
        self.A_timeVar = None
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

    def Apply(self):
        global Node, max_evaluations, population_size, departure_lon, arrival_lon, departure_lat, arrival_lat, lower_bound_v, lower_bound_t, upper_bound_v, upper_bound_t, del_t, draught
        self.lb = []
        self.ub = []
        self.lower_bound = str(self.lowerbound.text())
        self.upper_bound = str(self.upperbound.text())
        self.lb = self.lower_bound.split(',')
        self.ub = self.upper_bound.split(',')
        self.lower_bound_v = float(self.lb[0])
        self.lower_bound_t = float(self.lb[1])
        self.upper_bound_v = float(self.ub[0])
        self.upper_bound_t = float(self.ub[1])
        self.max_evaluations = int(self.maxevaluations.text())
        self.population_size = int(self.populationsize.text())
        self.Node = int(self.Node_line.text())
        self.del_t = 0
        self.draught = float(self.Draught_line.text())
        Node = self.Node
        lower_bound_v = self.lower_bound_v
        lower_bound_t = self.lower_bound_t
        upper_bound_v = self.upper_bound_v
        upper_bound_t = self.upper_bound_t
        max_evaluations = self.max_evaluations
        population_size = self.population_size
        del_t = self.del_t
        draught = self.draught
        self.D_dateVar = self.D_dateEdit.date()
        self.A_dateVar = self.A_dateEdit.date()
        self.D_timeVar = self.D_timeEdit.time()
        self.A_timeVar = self.A_timeEdit.time()
        strip_D_date = str(self.D_dateVar).strip('PyQt5.QtCore.QDate()')
        strip_D_time = str(self.D_timeVar).strip('PyQt5.QtCore.QTime()')
        strip_A_date = str(self.A_dateVar).strip('PyQt5.QtCore.QDate()')
        strip_A_time = str(self.A_timeVar).strip('PyQt5.QtCore.QTime()')
        print(strip_D_date, strip_D_time)
        print(strip_A_date, strip_A_time)
        D_date_split = strip_D_date.split(',')
        D_time_split = strip_D_time.split(',')
        A_date_split = strip_A_date.split(',')
        A_time_split = strip_A_time.split(',')
        D_time = []
        A_time = []
        for i in D_date_split:
            D_time.append(int(i))
        for i in A_date_split:
            A_time.append(int(i))
        for i in D_time_split:
            D_time.append(int(i))
        for i in A_time_split:
            A_time.append(int(i))
        print(tuple(D_time))
        print(tuple(A_time))
        D_time = tuple(D_time)
        A_time = tuple(A_time)
        time1 = datetime(int(D_time[0]), int(D_time[1]), int(D_time[2]), int(D_time[3]), int(D_time[4]))
        time2 = datetime(int(A_time[0]), int(A_time[1]), int(A_time[2]), int(A_time[3]), int(A_time[4]))
        print(time1)
        print(time2)
        #time1 = datetime(2020, 5, 17, 20, 0, 0)
        #time2 = datetime(2021, 5, 20, 10, 0, 2)
        time_diff = time2-time1
        print(time_diff)
        print((time_diff.days*24)+(time_diff.seconds/3600))
        self.del_t = int((time_diff.days*24)+(time_diff.seconds/3600))


        lat = (float(departure_lat) + float(arrival_lat)) / 2
        lon = (float(departure_lon) + float(arrival_lon)) / 2


        self.m = folium.Map(
            location=[lat, lon],
            tiles='Stamen Terrain',
            zoom_start=4
        )
        data = io.BytesIO()
        self.m.save(data, close_file=False)
        self.webView.setHtml(data.getvalue().decode())
        input_value = open('./resources/value.txt', 'w', encoding='utf-8')
        input_value.writelines(str(self.Node) + '\n')
        input_value.writelines(str(departure_lon) + '\n')
        input_value.writelines(str(arrival_lon) + '\n')
        input_value.writelines(str(departure_lat) + '\n')
        input_value.writelines(str(arrival_lat) + '\n')
        input_value.writelines(str(self.lower_bound_v) + '\n')
        input_value.writelines(str(self.lower_bound_t) + '\n')
        input_value.writelines(str(self.upper_bound_v) + '\n')
        input_value.writelines(str(self.upper_bound_t) + '\n')
        input_value.writelines(str(self.max_evaluations) + '\n')
        input_value.writelines(str(self.population_size) + '\n')
        input_value.writelines(str(self.del_t) + '\n')
        input_value.writelines(str(self.draught) + '\n')
        input_value.close()



        return

    def astar(self):
        print('start')
        self.threadclass2.start()
        departure = departure_lat, departure_lon
        arrival = arrival_lat, arrival_lon
        self.progressBar.reset()
        while (True):
            from Astar import node1
            if node1 == 0:
                count_g = 0
            else:
                count_g = int((100 / vincenty(departure, arrival)) * (vincenty(departure, arrival) - node1))
            self.progressBar.setValue(count_g)
            QApplication.processEvents()
            self.sleep(30)
            print(count_g)
            if count_g >= 99:
                self.progressBar.setValue(100)
                break
        while (True):
            from functions.Optimizer_run import TFOC
            if TFOC == 0:
                self.sleep(10)
                continue
            else:
                break
        self.sleep(2000)
        from functions.Optimizer_run import TFOC, Distance, px, py

        icon_image1 = './GUI/ship_boat_vessel_icon.png'  # departure
        icon1 = CustomIcon(icon_image1, icon_size=(40, 40))
        if self.text == 'Busan':
            folium.Marker(
                [py[0], px[0]],
                popup='<b>Busan</b>Port',
                icon=icon1, tooltip='Departure').add_to(self.m)

        elif self.text == 'HochiMinh':
            folium.Marker(
                [py[0], px[0]],
                popup='<b>CatLai</b>Port',
                icon=icon1, tooltip='Departure').add_to(self.m)

        icon_image2 = './GUI/port_anchor_icon.png'  # arrival
        icon2 = CustomIcon(icon_image2, icon_size=(40, 40))
        if self.text2 == 'HochiMinh':
            folium.Marker(
                [py[-1], px[-1]],
                popup='<b>CatLai</b>Port',
                icon=icon2, tooltip='Arrival').add_to(self.m)
        elif self.text2 == 'Manila':
            folium.Marker(
                [py[-1], px[-1]],
                popup='<b>Manila</b>Port',
                icon=icon2, tooltip='Arrival').add_to(self.m)
        elif self.text2 == 'Singapore':
            folium.Marker(
                [py[-1], px[-1]],
                popup='<b>Singapore</b>Port',
                icon=icon2, tooltip='Arrival').add_to(self.m)
        elif self.text2 == 'Busan':
            folium.Marker(
                [py[-1], px[-1]],
                popup='<b>Busan</b>Port',
                icon=icon2, tooltip='Arrival').add_to(self.m)
        lines_astar = []
        for i in range(len(px)):
            li = []
            li.append(py[i])
            li.append(px[i])
            lines_astar.append(li)
        folium.PolyLine(
            locations=lines_astar,
            color='#FFFFFF',
            tooltip='path'
        ).add_to(self.m)

        self.Astar_distance.clear()
        self.Astar_distance.setText('%0.4f' % Distance)

        data = io.BytesIO()
        self.m.save(data, close_file=False)
        self.webView.setHtml(data.getvalue().decode())

        self.Astar_TFOC.clear()
        self.Astar_TFOC.setText('%0.4f' % TFOC)

        print('end')

    def onActedate_departure(self, text):
        global departure_lat, departure_lon, arrival_lon, arrival_lat
        self.text = text
        if self.text == 'Busan':
            departure_lat = 35.046
            departure_lon = 129.627
        elif self.text == 'HochiMinh':
            departure_lat = 9.62
            departure_lon = 107.992
        return

    def onActedate_arrival(self, text):
        global departure_lat, departure_lon, arrival_lon, arrival_lat
        self.text2 = text
        if self.text2 == 'Busan':
            arrival_lat = 35.046
            arrival_lon = 129.627
        elif self.text2 == 'HochiMinh':
            arrival_lat = 9.62
            arrival_lon = 107.992
        elif self.text2 == 'Manila':
            arrival_lat = 14.583
            arrival_lon = 120.950
        elif self.text2 == 'Singapore':
            arrival_lat = 1.292
            arrival_lon = 103.725
        return

    def sleep(self,n):
        QtTest.QTest.qWait(n)

    def loadweather(self):
        global Wave_array, Node, max_evaluations, population_size, departure_lon, arrival_lon, departure_lat, arrival_lat, lower_bound_v, lower_bound_t, upper_bound_v, upper_bound_t
        # # polyon
        for k in range(50):
            for u in range(50):
                k_2 = k
                u_2 = u
                if Wave_array[k_2, u_2] <= 1:
                    pass
                elif Wave_array[k_2, u_2] < 10:
                    k1 = 40 - k_2
                    k2 = 39 - k_2
                    u1 = 100 + u_2
                    u2 = 101 + u_2
                    folium.Polygon(
                        locations=[(k1, u1), (k1, u2), (k2, u2), (k2, u1)],
                        fill=True,
                        tooltip='wave_Height-1m',
                        color='#0d93d6',
                        stroke=False
                    ).add_to(self.m)
                elif Wave_array[k_2, u_2] < 20:
                    k1 = 40 - k_2
                    k2 = 39 - k_2
                    u1 = 100 + u_2
                    u2 = 101 + u_2
                    folium.Polygon(
                        locations=[(k1, u1), (k1, u2), (k2, u2), (k2, u1)],
                        fill=True,
                        tooltip='wave_Height-2m',
                        color='#2af520',
                        stroke=False
                    ).add_to(self.m)
                elif Wave_array[k_2, u_2] < 30:
                    k1 = 40 - k_2
                    k2 = 39 - k_2
                    u1 = 100 + u_2
                    u2 = 101 + u_2
                    folium.Polygon(
                        locations=[(k1, u1), (k1, u2), (k2, u2), (k2, u1)],
                        fill=True,
                        tooltip='wave_Height-3m',
                        color='#edff2b',
                        stroke=False
                    ).add_to(self.m)
                elif Wave_array[k_2, u_2] < 40:
                    k1 = 40 - k_2
                    k2 = 39 - k_2
                    u1 = 100 + u_2
                    u2 = 101 + u_2
                    folium.Polygon(
                        locations=[(k1, u1), (k1, u2), (k2, u2), (k2, u1)],
                        fill=True,
                        tooltip='wave_Height-4m',
                        color='#f58320',
                        stroke=False
                    ).add_to(self.m)
                elif Wave_array[k_2, u_2] < 50:
                    k1 = 40 - k_2
                    k2 = 39 - k_2
                    u1 = 100 + u_2
                    u2 = 101 + u_2
                    folium.Polygon(
                        locations=[(k1, u1), (k1, u2), (k2, u2), (k2, u1)],
                        fill=True,
                        tooltip='wave_Height-5m',
                        color='#f53220',
                        stroke=False
                    ).add_to(self.m)
        data = io.BytesIO()
        self.m.save(data, close_file=False)
        self.webView.setHtml(data.getvalue().decode())

    def OPtimizer(self):
        global Node, max_evaluations, population_size, departure_lon, arrival_lon, departure_lat, arrival_lat, lower_bound_v, lower_bound_t, upper_bound_v, upper_bound_t
        self.threadclass.start()
        self.progressBar.reset()
        while (True):
            from functions.defined_problem import count
            count_g = int((100 / max_evaluations) * count)
            self.progressBar.setValue(count_g)
            QApplication.processEvents()
            self.sleep(10)
            if count_g >= 100:
                self.progressBar.setValue(100)
                break


        while(True):
            try:
                from functions.Optimizer_run import lines
                print(lines)
            except:
                self.sleep(10)
                continue
            else:
                break
        self.sleep(2000)
        from functions.Optimizer_run import x, y, obj, lines
        distance = 0
        distance_list = []
        velocity = []
        angle_list = []
        for i in range(len(x) - 1):
            a = (y[i], x[i])
            b = (y[i + 1], x[i + 1])
            distance += vincenty(a, b)
            distance_list.append(vincenty(a, b))
            velocity.append(distance_list[i] / (self.del_t * 1.852))
            angle_list.append(jinbuk(y[i], x[i], y[i + 1], x[i + 1]))


        folium.PolyLine(
            locations=lines,
            color='#ff4264',
            tooltip='path'
        ).add_to(self.m)
        for i, j in zip(x, y):
            folium.Circle(
                location=(j, i),
                radius=10000,
                color='yellow'
            ).add_to(self.m)

        icon_image1 = './GUI/ship_boat_vessel_icon.png'  # departure
        icon1 = CustomIcon(icon_image1, icon_size=(40, 40))
        if self.text2 == 'Busan':
            folium.Marker(
                [y[0], x[0]],
                popup='<b>Busan</b>Port',
                icon=icon1, tooltip='Departure').add_to(self.m)

        elif self.text2 == 'HochiMinh':
            folium.Marker(
                [y[0], x[0]],
                popup='<b>CatLai</b>Port',
                icon=icon1, tooltip='Departure').add_to(self.m)

        icon_image2 = './GUI/port_anchor_icon.png'  # arrival
        icon2 = CustomIcon(icon_image2, icon_size=(40, 40))
        if self.text2 == 'HochiMinh':
            folium.Marker(
                [y[-1], x[-1]],
                popup='<b>CatLai</b>Port',
                icon=icon2, tooltip='Arrival').add_to(self.m)
        elif self.text2 == 'Manila':
            folium.Marker(
                [y[-1], x[-1]],
                popup='<b>Manila</b>Port',
                icon=icon2, tooltip='Arrival').add_to(self.m)
        elif self.text2 == 'Singapore':
            folium.Marker(
                [y[-1], x[-1]],
                popup='<b>Singapore</b>Port',
                icon=icon2, tooltip='Arrival').add_to(self.m)
        elif self.text2 == 'Busan':
            folium.Marker(
                [y[-1], x[-1]],
                popup='<b>Busan</b>Port',
                icon=icon2, tooltip='Arrival').add_to(self.m)



        data = io.BytesIO()
        self.m.save(data, close_file=False)
        self.webView.setHtml(data.getvalue().decode())
        self.GA_Distance.clear()
        self.GA_Distance.setText('%0.4f' % distance)
        self.GA_TFOC.clear()
        self.GA_TFOC.setText('%0.4f' % (float(obj)))


if __name__ == "__main__":
    app = QApplication(sys.argv)
    myWindow = MyWindow()
    myWindow.show()
    app.exec_()


