from PyQt5 import QtCore
from jmetal.algorithm.multiobjective.nsgaii import NSGAII
from jmetal.operator import SBXCrossover, PolynomialMutation
from jmetal.util.comparator import DominanceComparator
from jmetal.util.solution import print_function_values_to_file, print_variables_to_file
from jmetal.util.termination_criterion import StoppingByEvaluations
from functions.defined_problem import Linear4
from vincenty import vincenty
import functions.jin_buk_theta
import numpy as np
import pandas as pd
from functions.vincenty_direct import vincenty_direct

lines = []
x = []
y = []
obj = 0
TFOC = 0
Distance = 0
px = []
py = []

class ThreadClass(QtCore.QThread):

    def run(self):
        global x, y, lines,obj
        valuelist = []
        valuelist_in = []
        get_value = open('./resources/value.txt', 'r', encoding='utf-8')
        for i in range(13):
            value = get_value.readline()
            valuelist.append(value)
            value_real = valuelist[i].strip()
            valuelist_in.append(value_real)
            valuelist_in[i] = float(valuelist_in[i])
        print(valuelist_in)
        get_value.close()
        self.Node = valuelist_in[0]
        self.lower_bound_v = valuelist_in[5]
        self.lower_bound_t = valuelist_in[6]
        self.upper_bound_v = valuelist_in[7]
        self.upper_bound_t = valuelist_in[8]
        self.max_evaluations = valuelist_in[9]
        self.population_size = valuelist_in[10]
        self.departure_lon = valuelist_in[1]
        self.departure_lat = valuelist_in[3]
        self.arrival_lon = valuelist_in[2]
        self.arrival_lat = valuelist_in[4]
        self.del_t = valuelist_in[11]
        self.draught = valuelist_in[12]

        print(type(self.Node))
        Node = int(self.Node)
        lower_bound_v = self.lower_bound_v
        lower_bound_t = self.lower_bound_t
        upper_bound_v = self.upper_bound_v
        upper_bound_t = self.upper_bound_t
        max_evaluations = int(self.max_evaluations)
        population_size = int(self.population_size)
        departure_lon = self.departure_lon
        departure_lat = self.departure_lat
        arrival_lon = self.arrival_lon
        arrival_lat = self.arrival_lat
        del_t = int(self.del_t)
        draught = self.draught

        #print(Node)

        problem = Linear4(Node, departure_lon, arrival_lon, departure_lat, arrival_lat,
                          lower_bound_v, lower_bound_t, upper_bound_v, upper_bound_t, del_t, draught)

        algorithm = NSGAII(
            problem=problem,
            population_size=population_size,
            offspring_population_size=100,
            mutation=PolynomialMutation(probability=1.0 / problem.number_of_variables, distribution_index=20.0),
            crossover=SBXCrossover(probability=0.9, distribution_index=20.0),
            termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations),
            dominance_comparator=DominanceComparator()
        )

        algorithm.run()

        front = algorithm.get_result()

        # Save results to file
        print_function_values_to_file(front, 'FUN3_부산_호치민_(node=40)_new_model_FOC_날씨테스트(밑).' + algorithm.get_name() + "-" + problem.get_name())
        print_variables_to_file(front, 'VAR3_linear_부산_호치민_(node=40)_new_model_FOC_날씨테스트(밑).' + algorithm.get_name() + "-" + problem.get_name())

        file_open = open('VAR3_linear_부산_호치민_(node=40)_new_model_FOC_날씨테스트(밑).' + algorithm.get_name() + "-" + problem.get_name(), 'r',
                         encoding='utf-8')

        file_open2 = open('FUN3_부산_호치민_(node=40)_new_model_FOC_날씨테스트(밑).' + algorithm.get_name() + "-" + problem.get_name(), 'r',
                          encoding='utf-8')
        obj_line = file_open2.readlines()
        obj = obj_line[0]

        file_open2.close()

        line = file_open.readlines()
        word = line[0].split(" ")

        distance_list = []
        distance_list2 = []

        v_list = []
        '''
        total_v = 0
        self.del_t_list = []
        for i in range(int(self.Node / 2) - 1):
            v_list.append(float(word[i*2]))
        for i in v_list:
            total_v += i


        v_sort = sorted(v_list)
        v_sort_r = sorted(v_list, reverse=True)

        for i in range(len(v_list) // 2):
            k = v_list.index(v_sort[i])
            v_list[k] = v_sort_r[i]
            r = v_list.index(v_sort_r[i])
            v_list[r] = v_sort[i]

        for i in v_list:
            time = (i / total_v) * 115
            self.del_t_list.append(time)
        '''

        for i in range(int(self.Node / 2) - 1): # -1 뺌
            distance = float(word[i*2]) * 6 * 1000
            distance_list.append(distance)

        x.append(departure_lon)
        y.append(departure_lat)
        for i in range(int(self.Node / 2) - 1):
            next = vincenty_direct(y[i], x[i], float(word[2 * i + 1]), int(distance_list[i]))
            x.append(next[1])
            y.append(next[0])


        x.append(arrival_lon)
        y.append(arrival_lat)

        total_time = 0
        v_list = []
        total_v = 0
        del_t_list = []
        for i in range(int(self.Node / 2)):
            v_list.append(float(word[2 * i]))






        for i in range(len(x)):
            print(x[i], y[i])


        for i in range(len(x)-1):
            distance = vincenty((y[i],x[i]), (y[i+1], x[i+1]))
            distance_list2.append(distance)


        for i in range(len(v_list)):
            del_t_list.append(distance_list2[i]/v_list[i])
            total_time += del_t_list[i]

        print(del_t_list)
        print(total_time)




        for i in range(len(x)):
            li = []
            li.append(y[i])
            li.append(x[i])
            lines.append(li)

        print('Algorithm (continuous problem): ' + algorithm.get_name())
        print('Problem: ' + problem.get_name())
        print('Computing time: ' + str(algorithm.total_computing_time))
        file_open.close()
        return

class ThreadClass2(QtCore.QThread):

    def run(self):
        valuelist = []
        valuelist_in = []
        get_value = open('./resources/value.txt', 'r', encoding='utf-8')
        for i in range(13):
            value = get_value.readline()
            valuelist.append(value)
            value_real = valuelist[i].strip()
            valuelist_in.append(value_real)
            valuelist_in[i] = float(valuelist_in[i])

        get_value.close()
        self.departure_lon = valuelist_in[1]
        self.departure_lat = valuelist_in[3]
        self.arrival_lon = valuelist_in[2]
        self.arrival_lat = valuelist_in[4]
        self.draught = valuelist_in[12]


        departure_lon = self.departure_lon
        departure_lat = self.departure_lat
        arrival_lon = self.arrival_lon
        arrival_lat = self.arrival_lat
        import Astar
        global Distance, TFOC, py, px
        Astar.main(departure_lon, departure_lat, arrival_lat, arrival_lon, self.draught)
        print(Astar.Px)
        for i in range(int((len(Astar.Py) - 1)/2)):
            py.append(Astar.Py[2 * i])
            px.append(Astar.Px[2 * i])

        angle_list = []
        distance_list = []
        distance_list_first = []
        speed = 15
        time = []

        for i in range(len(Astar.Px) - 1):
            a = (Astar.Py[i], Astar.Px[i])
            b = (Astar.Py[i + 1], Astar.Px[i + 1])
            distance_list_first.append(vincenty(a, b))

        for i in range(int((len(distance_list_first) / 2) - 1)):
            distance = distance_list_first[i] + distance_list_first[i + 1]
            distance_list.append(distance)
            Distance += distance

        for i in range(len(distance_list)):
            time.append(distance_list[i] / (speed * 1.825))

        for i in range(len(px) - 1):
            angle_astar = functions.jin_buk_theta.jinbuk(py[i], px[i], py[i + 1], px[i + 1])
            angle_list.append(angle_astar)

        df_frame_Astar = pd.read_excel('./Dataset/test_astar.xlsx', header=0, sheet_name=None, engine='openpyxl')
        input_df_Astar = df_frame_Astar['FOC']

        Draught = self.draught
        for i in range(len(distance_list)):
            distance1 = + distance_list[i]
            times = int(((distance1 / (speed * 1.852)) / 6) + 1)
            data = Astar.df['Weatherdata%d' % times]
            df2 = data.set_index('latitude & longitude')
            input_df_Astar.iloc[[i],[0]] = speed
            input_df_Astar.iloc[[i],[1]] = Draught
            input_df_Astar.iloc[[i],[2]] = angle_list[i]
            input_df_Astar.iloc[[i],[3]] = df2.loc['{}:{}'.format(int(py[i]), int(px[i])), 'Wind_speed']
            input_df_Astar.iloc[[i],[4]] = df2.loc['{}:{}'.format(int(py[i]), int(px[i])), 'Wind_Direction']
            input_df_Astar.iloc[[i],[5]] = df2.loc['{}:{}'.format(int(py[i]), int(px[i])), 'Wave_Height']
            input_df_Astar.iloc[[i],[6]] = df2.loc['{}:{}'.format(int(py[i]), int(px[i])), 'Wave_Direction']
            input_df_Astar.iloc[[i],[7]] = df2.loc['{}:{}'.format(int(py[i]), int(px[i])), 'Wave_Frequency']
            input_df_Astar.iloc[[i],[8]] = time[i]

        featureList = ['Speed2', 'Draught', 'Course', 'WindSpeed', 'WindDirectionDeg', 'WaveHeight', 'WaveDirection',
                       'WavePeriod','Time']
        trainContinuous = Astar.scaler.transform(input_df_Astar[featureList])
        testX = np.hstack([trainContinuous])
        preds = Astar.model.predict(testX)
        Foc_Time = preds.flatten()
        Foc = []
        for i in range(len(time)):
            Foc.append((Foc_Time[i] * time[i]))

        for i in range(len(Foc)):
            input_df_Astar.iloc[[i], [8]] = Foc[i]

        for i in range(len(distance_list)):
            TFOC += Foc[i]

        print('end')
        return


