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

        #solution_list[0].variables = [13.428173317280072, -2.2482751065172337, 13.938174945668166, -2.244376412346751, 13.799045930555502, -2.246634162040095, 13.175153834452312, -2.243146386852296, 13.918208825656164, -2.242961691829457, 13.425785953460062, -2.246757198450582,
                                      #13.382050937502862, -2.242896225234722, 13.124710333609992, 3.1415925851265736, 14.180524626017583, -2.2441828998559665, 14.06119411221475, -2.247073102352685, 13.7299635299142, -2.24309276575986, 14.588029072565433, -2.2458566951724945,
                                      #13.54389371109228, -2.2396569572614897, 13.58863646386737, -2.2445473743750903, 13.370046095858747, -2.249992351450045, 14.03189362113152, -2.24494496658679, 14.08719193054638, -2.2411341087160563, 14.492114271303468, -2.2462088819599795,
                                      #14.164421430542353, -2.2469900346263794, 13.144885001296856, 3.1150615011080465]
        solution_list[0].variables = [6.365395631543483, -2.2458498980974686, 6.523190332626875, -2.245178303699748, 6.69620945985657, -2.241293789019776, 6.298519935706621, -2.2461958861747577, 6.524801941270208, -2.2477798942871363, 6.640919469183086, -2.247229700378625,
                                      6.789897872333004, -2.2488201892331308, 6.54415534848224, -2.239284209525101, 6.246222032465558, 3.1415919054491246, 6.381431837472593, -2.2434527653893355, 6.212799498057493, 3.1415926422749747, 6.561825158598168, -2.2481263578694275,
                                      6.611902538893728, -2.2479680140021006, 6.39891747106046, -2.2465782913665184, 6.639271777636226, -2.244786454348959, 6.633218163066058, -2.2445500193605477, 6.464413305144251, -2.243939389240185, 6.60016828660881, -2.245973566738533,
                                      6.938988910552903, -2.247009815511887, 6.248190811639581, 1.9686137093522176]

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
