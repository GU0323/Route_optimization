from jmetal.algorithm.singleobjective.genetic_algorithm import GeneticAlgorithm
from jmetal.operator import BinaryTournamentSelection, PolynomialMutation, SBXCrossover
from jmetal.problem.singleobjective.unconstrained import Rastrigin
from jmetal.util.termination_criterion import StoppingByEvaluations

if __name__ == '__main__':
    problem = Rastrigin(10)

    algorithm = GeneticAlgorithm(
        problem=problem,
        population_size=100, #부모님 수
        offspring_population_size=100, # 자식 수
        mutation=PolynomialMutation(1.0 / problem.number_of_variables, 20.0), #20 = distribution index = 특정한 operator(연산자)?
        crossover=SBXCrossover(0.9, 20.0), # 단일 점 돌연변이 연산자는 돌연변이 확률만 요구하지만 SBX crossover는 확률과 distribution index 요구
        selection=BinaryTournamentSelection(), # 선택 함수
        termination_criterion=StoppingByEvaluations(max_evaluations=500000) # 최적화 횟수? 계산 횟수?
    )

    algorithm.run()
    result = algorithm.get_result()

    print('Algorithm: {}'.format(algorithm.get_name()))
    print('Problem: {}'.format(problem.get_name()))
    print('Solution: {}'.format(result.variables))
    print('Fitness: {}'.format(result.objectives[0]))
    print('Computing time: {}'.format(algorithm.total_computing_time))
