import numpy as np
from math import sqrt, cos, sin
from jmetal.core.problem import FloatProblem
from jmetal.core.solution import FloatSolution
from vincenty import vincenty
import pandas as pd
from tensorflow.keras.models import load_model
import joblib
from functions.jin_buk_theta import jinbuk
from functions.vincenty_direct import vincenty_direct

maze = np.load('./resources/mapImage/maparray.npy')
count = 0
df = pd.read_excel('./Dataset/Weather_information2.xlsx')
df_frame = pd.read_excel('./Dataset/testxl.xlsx', header=0, sheet_name=None, engine='openpyxl')
input_df = df_frame['FOC']
model = load_model("./Models/FOCModels14-1")
scaler_filename = "./Models/FOCModels14-1" + "/scaler.save"
scaler = joblib.load(scaler_filename)

class Linear4(FloatProblem):

    def __init__(self, number_of_variables, x1, x2, y1, y2, lowerbound_v, lowerbound_t, upperbound_v, upperbound_t,del_t):
        super(Linear4, self).__init__()
        global df,input_df
        self.number_of_objectives = 1
        self.number_of_variables = number_of_variables
        self.number_of_constraints = int(number_of_variables / 2) + 1
        self.obj_directions = [self.MINIMIZE]
        self.lower_bound = [lowerbound_v, lowerbound_t] * (int(number_of_variables / 2))
        self.upper_bound = [upperbound_v, upperbound_t] * (int(number_of_variables / 2))
        self.Px_departure = x1
        self.Py_departure = y1
        self.Px_arrival = x2
        self.Py_arrival = y2
        self.del_t = int(del_t)
        self.x = []
        self.y = []
        self.del_t_list = []
        self.del_t_list2 = []

    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        global df, input_df, model, scaler
        TFOC = 0
        S = solution.variables
        distance_list = []
        v_list = []
        total_v = 0
        self.del_t_list = []
        for i in range(int(self.number_of_variables / 2) - 1):
            v_list.append(S[2 * i])
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


        for i in range(int(self.number_of_variables / 2) - 1): # -1 뺌
            distance = S[2 * i] * self.del_t_list[i] * 1000
            distance_list.append(distance)

        self.x = []
        self.y = []
        self.x.append(self.Px_departure)
        self.y.append(self.Py_departure)
        for i in range(int(self.number_of_variables / 2) - 1):
            next = vincenty_direct(self.y[i], self.x[i], S[2 * i + 1], distance_list[i])
            self.x.append(next[1])
            self.y.append(next[0])

        self.x.append(self.Px_arrival)
        self.y.append(self.Py_arrival)
        last_distance = vincenty((self.y[-2], self.x[-2]), (self.y[-1], self.x[-1]))
        last_distance = last_distance * 1000
        last_angle = jinbuk(self.y[-2], self.x[-2], self.y[-1], self.x[-1])
        angle_list = []
        for i in range(int((self.number_of_variables/2) - 1)):
            angle_list.append(S[i*2+1])


        angle_list.append(last_angle)
        self.del_t_list.append((last_distance/1000)/S[-2])



        distance_list.append(last_distance)

        knot_velocity = []
        for i in range(int(self.number_of_variables/2)):
            knot_velocity.append(S[i*2]/1.852)


        df2 = df.set_index('latitude & longitude')
        evalue_weather = 0
        for i in range(len(self.x) - 1):
            if int(self.x[i]) > 100 and int(self.x[i]) < 150 and int(self.y[i]) > -9 and int(self.y[i]) < 41:
                evalue_weather += 0
            else:
                evalue_weather += 1
        Draught = 15
        if evalue_weather == 0:
            for k in range(len(self.x) - 1):
                input_df.iloc[[k], [0]] = knot_velocity[k]
                input_df.iloc[[k], [1]] = Draught
                input_df.iloc[[k], [2]] = angle_list[k]
                input_df.iloc[[k], [3]] = df2.loc['{}:{}'.format(int(self.y[k]), int(self.x[k])), 'Wind_speed']
                input_df.iloc[[k], [4]] = df2.loc['{}:{}'.format(int(self.y[k]), int(self.x[k])), 'Wind_Direction']
                input_df.iloc[[k], [5]] = df2.loc['{}:{}'.format(int(self.y[k]), int(self.x[k])), 'Wave_Height']
                input_df.iloc[[k], [6]] = df2.loc['{}:{}'.format(int(self.y[k]), int(self.x[k])), 'Wave_Direction']
                input_df.iloc[[k], [7]] = df2.loc['{}:{}'.format(int(self.y[k]), int(self.x[k])), 'Wave_Frequency']
                input_df.iloc[[k], [8]] = self.del_t_list[k] / self.del_t_list[k]

            featureList = ['Speed2', 'Draught', 'Course', 'WindSpeed', 'WindDirectionDeg', 'WaveHeight', 'WaveDirection', 'WavePeriod', 'Time']

            print(input_df)

            trainContinuous = scaler.transform(input_df[featureList])
            testX = np.hstack([trainContinuous])

            preds = model.predict(testX)
            Foc = preds.flatten()

            FOC = Foc.tolist()
            print(FOC)
            TFOC_list = []



            for i in range(len(FOC)):
                TFOC += FOC[i] * self.del_t_list[i]
                TFOC_list.append(FOC[i] * self.del_t_list[i])
            print(TFOC_list)
            print(TFOC)

        else:
            TFOC += 10000
        '''
        total_distance = 0
        RSME = 0
        for i in range(len(distance_list)):
            total_distance += distance_list[i]
        
        for i in range(len(distance_list)):
            R_mean = total_distance / int(self.number_of_variables / 2)
            RSME += (distance_list[i] - R_mean) ** 2

        R_diviation = sqrt(RSME / int(self.number_of_variables / 2))
        '''
        solution.objectives[0] = TFOC
        self.evaluate_constraints(solution)

        return solution

    def evaluate_constraints(self, solution: FloatSolution) -> None:
        constraints = [0.0 for _ in range(self.number_of_constraints)]
        x2 = []
        y2 = []
        time_max = 120
        total_time = 0

        for i in range(len(self.del_t_list)):
            total_time += self.del_t_list[i]


        for i in range(len(self.x)):
            A = 1799 + self.x[i] * 10
            B = 899 + self.y[i] * 10
            x2.append(A)
            y2.append(B)

        for i in range(int(solution.number_of_variables / 2)):
            between_node_distance_x = sqrt((x2[i + 1] - x2[i]) ** 2)
            between_node_distance_y = sqrt((y2[i + 1] - y2[i]) ** 2)
            cp_x = 0
            cp_y = 0
            if x2[i] > x2[i + 1]:
                cp_x = x2[i + 1]
            elif x2[i] < x2[i + 1]:
                cp_x = x2[i]
            else:
                constraints[i] = -1

            if y2[i] > y2[i + 1]:
                cp_y = y2[i + 1]
            elif y2[i] < y2[i + 1]:
                cp_y = y2[i]
            else:
                constraints[i] = -1
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

            if maze[int(1799 - y2[i + 1])][int(x2[i + 1])] == 1 or maze[int(1799 - y2[i + 1] + 1)][
                int(x2[i + 1]) + 1] == 1 or maze[int(1799 - y2[i + 1] - 1)][int(x2[i + 1]) - 1] == 1 \
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
                constraints[i] = -10


        constraints[int(solution.number_of_variables/2)] = time_max - total_time

        global count
        count += 1
        print(count)

        solution.constraints = constraints
        return

    def get_name(self) -> str:
        return 'Linear4'




