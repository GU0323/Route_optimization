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

        solution_list[0].variables = [2.847082993449695, -2.136920176808673, 2.848356683328698, -2.1274763933014076, 2.8487046346435485, -2.2039814881144593, 2.846346058488574, -1.7104228541826305,
                                      2.84221207821379, -1.9550640254533171, 2.8466379788403344, -1.858506863698536 ,2.854732471555902, -2.1148672960412886 ,2.8490518638573707, -2.447218657532797,
                                      2.8466295481258226, -2.3634565203248945, 2.84349449436732, -2.3859417900060738, 2.8467109970510633, -2.4136002325416874, 2.846926205938669, -2.423916570650629,
                                      2.8482550695668007, -2.406647615954939, 2.8417846714516526, -2.413688852510062, 2.849296676176231, -2.383003504099558, 2.841385498068435, -2.3995533144445753,
                                      2.8479222958578934, -2.4428728147219925, 2.842757260518458, -2.413336469297997, 2.8494445774928048, -2.4427390606672943, 1.6513017774058156, -2.4427392089319255]

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
