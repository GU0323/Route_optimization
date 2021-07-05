import functools
from abc import ABC, abstractmethod
from multiprocessing.pool import ThreadPool, Pool
from typing import TypeVar, List, Generic

try:
    import dask
except ImportError:
    pass

try:
    from pyspark import SparkConf, SparkContext
except ImportError:
    pass

from jmetal.core.problem import Problem

S = TypeVar('S')


class Evaluator(Generic[S], ABC):

    @abstractmethod
    def evaluate(self, solution_list: List[S], problem: Problem) -> List[S]:
        pass

    @staticmethod
    def evaluate_solution(solution: S, problem: Problem) -> None:
        problem.evaluate(solution)


class SequentialEvaluator(Evaluator[S]):

    def evaluate(self, solution_list: List[S], problem: Problem) -> List[S]:
        # solution_list[0].variables = [12.896330662823618, 0.6526531332031992, 12.896250361759966, 0.6904596777352663, 12.896567886507124, 0.59350453834493185, 12.896157857861037, 0.000449505792210343, 12.896103326426209, -0.4440885148216845, 12.896083227615273, -0.4451980244733752, 12.896360273516719, 1.5555667465428427]
        # solution_list[0].variables = [4.896330662823618, 0.5526531332031992, 4.896250361759966, 0.5904596777352663, 4.896567886507124, 0.49350453834493185, 4.896157857861037, 0.000449505792210343, 4.896103326426209, -0.5440885148216845, 4.896083227615273, -0.5451980244733752, 4.896360273516719, 1.5555667465428427]
        solution_list[0].variables = [10.480187052614321, 0.3458146867848035, 10.488809757208152, 0.48020942750301276,
                                      10.486492684337154, 0.7663015430621156, 10.482284623189209, 0.6338611744240933,
                                      10.479962046886006, 0.6252019408761804, 10.480277588168347, 0.7263638930117486,
                                      10.480871145273714, 0.5318469502755498, 10.479816139208622, 1.2621617632147928,
                                      10.479824160041883, 1.2527021483955836, 10.484054924793907, 0.30830870190391635]
        for solution in solution_list:
            Evaluator.evaluate_solution(solution, problem)

        return solution_list


class MapEvaluator(Evaluator[S]):

    def __init__(self, processes: int = None):
        self.pool = ThreadPool(processes)

    def evaluate(self, solution_list: List[S], problem: Problem) -> List[S]:
        self.pool.map(lambda solution: Evaluator.evaluate_solution(solution, problem), solution_list)

        return solution_list


class MultiprocessEvaluator(Evaluator[S]):
    def __init__(self, processes: int = None):
        super().__init__()
        self.pool = Pool(processes)

    def evaluate(self, solution_list: List[S], problem: Problem) -> List[S]:
        return self.pool.map(functools.partial(evaluate_solution, problem=problem), solution_list)


class SparkEvaluator(Evaluator[S]):
    def __init__(self, processes: int = 8):
        self.spark_conf = SparkConf().setAppName("jmetalpy").setMaster(f"local[{processes}]")
        self.spark_context = SparkContext(conf=self.spark_conf)

        logger = self.spark_context._jvm.org.apache.log4j
        logger.LogManager.getLogger("org").setLevel(logger.Level.WARN)

    def evaluate(self, solution_list: List[S], problem: Problem) -> List[S]:
        solutions_to_evaluate = self.spark_context.parallelize(solution_list)

        return solutions_to_evaluate \
            .map(lambda s: problem.evaluate(s)) \
            .collect()


def evaluate_solution(solution, problem):
    Evaluator[S].evaluate_solution(solution, problem)
    return solution


class DaskEvaluator(Evaluator[S]):
    def __init__(self, scheduler='processes'):
        self.scheduler = scheduler

    def evaluate(self, solution_list: List[S], problem: Problem) -> List[S]:
        with dask.config.set(scheduler=self.scheduler):
            return list(dask.compute(*[
                dask.delayed(evaluate_solution)(solution=solution, problem=problem) for solution in solution_list
            ]))
