from vincenty import vincenty
from tensorflow.keras.models import load_model
import numpy as np
import functions.jin_buk_theta
import pandas as pd
import joblib


df = pd.read_excel('./Dataset/Weather_information2.xlsx',  header=0, sheet_name=None, engine='openpyxl')
df_frame = pd.read_excel('./Dataset/test_astar.xlsx', header=0, sheet_name=None, engine='openpyxl')
input_df = df_frame['FOC']
model = load_model("./Models/consumption_model_42000-2")
scaler_filename = "./Models/consumption_model_42000-2" + "/scaler.save"
scaler = joblib.load(scaler_filename)
node1 = 0

class Node:
    def __init__(self, parent=None, position=None):
        self.parent = parent
        self.position = position

        self.g = 0
        self.h = 0
        self.f = 0

    def __eq__(self, other):
        return self.position == other.position


def heuristic(node, goal, D=1, D2=2 ** 0.5):  # Diagonal Distance
    dx = abs(node.position[0] - goal.position[0])
    dy = abs(node.position[1] - goal.position[1])
    return D * (dx + dy) + (D2 - 2 * D) * min(dx, dy)


def aStar(maze, start, end, Draught):
    global df, input_df, model, scaler, aaa
    # startNode와 endNode 초기화
    startNode = Node(None, start)
    endNode = Node(None, end)

    # openList, closedList 초기화
    openList = []
    closedList = []

    # openList에 시작 노드 추가
    openList.append(startNode)

    # endNode를 찾을 때까지 실행
    while openList:

        # 현재 노드 지정
        currentNode = openList[0]
        currentIdx = 0

        # 이미 같은 노드가 openList에 있고, f 값이 더 크면
        # currentNode를 openList안에 있는 값으로 교체
        for index, item in enumerate(openList):
            if item.f < currentNode.f:
                currentNode = item
                currentIdx = index

        # openList에서 제거하고 closedList에 추가
        openList.pop(currentIdx)
        closedList.append(currentNode)

        # 현재 노드가 목적지면 current.position 추가하고
        # current의 부모로 이동
        if currentNode == endNode:
            path = []
            current = currentNode
            while current is not None:
                # maze 길을 표시하려면 주석 해제
                # x, y = current.position
                # maze[x][y] = 7
                path.append(current.position)
                current = current.parent
            return path[::-1]  # reverse

        children = []
        # 인접한 xy좌표 전부
        for newPosition in [(-2, 2),(2, 2), (2, -2), (-2, -2),(-1, 2) , (0, 2), (1, 2), (-2, -1), (2, -1), (-1, -2), (0, -2), (1, -2), (2, 0), (-2, 0), (2, 1), (-2, 1)]:

            # 노드 위치 업데이트
            nodePosition = (
                currentNode.position[0] + newPosition[0],  # X
                currentNode.position[1] + newPosition[1])  # Y

            # 미로 maze index 범위 안에 있어야함
            within_range_criteria = [
                nodePosition[0] > (len(maze) - 1),
                nodePosition[0] < 0,
                nodePosition[1] > (len(maze[len(maze) - 1]) - 1),
                nodePosition[1] < 0,
            ]

            if any(within_range_criteria):  # 하나라도 true면 범위 밖임
                continue

            # 장애물이 있으면 다른 위치 불러오기
            if maze[nodePosition[0]][nodePosition[1]] == 0:
                continue

            new_node = Node(currentNode, nodePosition)
            children.append(new_node)
        child_list = []
        for child in children:

            # 자식이 closedList에 있으면 continue
            if child in closedList:
                continue
            child_list.append(child.position)
        draught = Draught
        for i in range(len(child_list)):
            departure = -((startNode.position[0] - 399) / 10), ((startNode.position[1] + 1001) / 10)
            start1 = -((currentNode.position[0] - 399) / 10), ((currentNode.position[1] + 1001) / 10)
            end1 = -((child_list[i][0] - 399) / 10), ((child_list[i][1] + 1001) / 10)
            arrival = -((endNode.position[0] - 399) / 10), ((endNode.position[1] + 1001) / 10)
            angle = functions.jin_buk_theta.jinbuk(start1[0], start1[1], end1[0], end1[1])
            times = int(((vincenty(departure, arrival) - vincenty(end1, arrival)) / (16.5 * 1.852)) / 6) +1
            if times < 1:
                first_data = df['Weatherdata1']
            elif times <=24 :
                first_data = df['Weatherdata%d' % times]
            else:
                first_data = df['Weatherdata24']
            df2 = first_data.set_index('latitude & longitude')
            input_df.iloc[[i], [0]] = 16.5
            input_df.iloc[[i], [1]] = draught
            input_df.iloc[[i], [2]] = angle
            input_df.iloc[[i], [3]] = df2.loc['{}:{}'.format(int(end1[0]), int(end1[1])), 'Wind_speed']
            input_df.iloc[[i], [4]] = df2.loc['{}:{}'.format(int(end1[0]), int(end1[1])), 'Wind_Direction']
            input_df.iloc[[i], [5]] = df2.loc['{}:{}'.format(int(end1[0]), int(end1[1])), 'Wave_Height']
            input_df.iloc[[i], [6]] = df2.loc['{}:{}'.format(int(end1[0]), int(end1[1])), 'Wave_Direction']
            input_df.iloc[[i], [7]] = df2.loc['{}:{}'.format(int(end1[0]), int(end1[1])), 'Wave_Frequency']
            input_df.iloc[[i], [8]] = 1

        featureList = ['Speed2', 'Draught', 'Course', 'WindSpeed', 'WindDirectionDeg', 'WaveHeight', 'WaveDirection',
                           'WavePeriod', 'Time']
        trainContinuous = scaler.transform(input_df[featureList])
        testX = np.hstack([trainContinuous])
        preds = model.predict(testX)
        i = -1
        # 자식들 모두 loop
        for child in children:
            # 자식이 closedList에 있으면 continue
            if child in closedList:
                continue
            i += 1
            # f, g, h값 업데이트
            end1 = -((child.position[0] - 399) / 10), ((child.position[1] + 1001) / 10)
            start1 = -((currentNode.position[0] - 399) / 10), ((currentNode.position[1] + 1001) / 10)
            arrival = -((endNode.position[0] - 399) / 10), ((endNode.position[1] + 1001) / 10)

            distance = vincenty(start1, end1) * 2
            time = distance / (16.5 * 1.825)
            child.g = (currentNode.g + abs(preds.flatten()[i] * time))
            child.h = vincenty(end1, arrival) / 5
            global node1
            node1 = vincenty(start1, arrival)
            child.f = child.g + child.h
            # 자식이 openList에 있으고, g값이 더 크면 continue
            if len([openNode for openNode in openList
                    if child == openNode and child.g > openNode.g]) > 0:
                continue
            openList.append(child)


Px = []
Py = []
def main(departure_lon, departure_lat, arrival_lat, arrival_lon, draught):

    # 1은 장애물

    maze = np.load('./resources/mapImage/local_map.npy')
    #부산 출발
    start_lat = departure_lon
    start_lon = departure_lat
    # 도착
    end_lat = arrival_lon
    end_lon = arrival_lat

    start = -int((int(start_lon*10) -399)), int((int(start_lat*10) - 1001))
    end = -int((int(end_lon*10) -399)), int((int(end_lat*10) - 1001))

    Draught = draught
    path = aStar(maze, start, end, Draught)

    for i in range(len(path)):
        py = -((path[i][0]-399) / 10)
        px = ((path[i][1]+1001) / 10)
        Px.append(px)
        Py.append(py)

    return Px, Py

if __name__ == '__main__':
    main()
    # [(0, 0), (1, 1), (2, 2), (3, 3), (4, 3), (5, 4), (6, 5), (7, 6)]
