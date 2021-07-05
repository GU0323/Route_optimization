from jmetal.algorithm.multiobjective.smpso import SMPSO
from jmetal.operator import PolynomialMutation
from jmetal.problem import ZDT4 #ZDT = Zitzler-Deb-Thiele 일반적으로 사용되는 벤치마크 문제
from jmetal.util.archive import CrowdingDistanceArchive
from jmetal.util.solution import print_function_values_to_file, print_variables_to_file
from jmetal.util.solution import read_solutions
from jmetal.util.termination_criterion import StoppingByEvaluations

if __name__ == '__main__':
    problem = ZDT4()
    problem.reference_front = read_solutions(filename='resources/reference_front/ZDT4.pf')
    #f1함수는 첫번째 결정 변수의 전용 함수일 뿐이고, g는 나머지 m-1변수의 함수이며, h의 매개변수는 f1과g의 함수 값을 가진다.
    # m = 10, x1 =[0,1] x2...xm[-5, 5], 전반적인 파레토 최적프론트는 g(x)=1.25일때 형성됨,


    max_evaluations = 25000
    algorithm = SMPSO(
        problem=problem,
        swarm_size=100,
        mutation=PolynomialMutation(probability=1.0 / problem.number_of_variables, distribution_index=20),
        leaders=CrowdingDistanceArchive(100),
        termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations)
    ) #population(swarm 사이즈는 100을 보통 모든 알고리즘들이 사용하고, AbYSS의 경우 기준 모집의 경우 20을 사용한다.)
    # 교차 연산자와 돌연변이 연산자의 분포지수는 20을 보통 사용한다.(distribution_index),
    #probability=1.0 / problem.number_of_variables = 돌연변이 확률을 의미한다.
    #PolynomialMutation => SMPSO에서는 다항식 돌연변이를 사용한다. OMOPSO는 uniformMutation을 사용

    algorithm.run()
    front = algorithm.get_result()

    # Save results to file
    print_function_values_to_file(front, 'FUN.' + algorithm.label)
    print_variables_to_file(front, 'VAR.'+ algorithm.label)

    print(f'Algorithm: ${algorithm.get_name()}')
    print(f'Problem: ${problem.get_name()}')
    print(f'Computing time: ${algorithm.total_computing_time}')
