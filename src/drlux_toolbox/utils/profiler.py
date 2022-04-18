import functools
import time
from loguru import logger
import tracemalloc


EXPERIMENT_FOLDER_PATH = "profiler.txt"

#use @time_profiler on function to invoke the profiler
def time_profiler(func):
    #logger.add(EXPERIMENT_FOLDER_PATH, filter=make_filter("profiler"))    
    logger.add(EXPERIMENT_FOLDER_PATH, filter=lambda record: record["extra"].get("name") == "profiler")    
    profiler_logger = logger.bind(name="profiler")
    
    """Print the runtime of the decorated function"""
    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        start_time = time.perf_counter()    # 1
        value = func(*args, **kwargs)
        end_time = time.perf_counter()      # 2
        run_time = end_time - start_time    # 3
        profiler_logger.info(f"Finished {func.__name__!r} in {run_time:.4f} secs")
        return value
    return wrapper_timer


def memory_profiler(func):
    #logger.add(EXPERIMENT_FOLDER_PATH, filter=make_filter("profiler"))    
    logger.add(EXPERIMENT_FOLDER_PATH, filter=lambda record: record["extra"].get("name") == "profiler")    
    profiler_logger = logger.bind(name="profiler")
    
    """Print the runtime of the decorated function"""
    @functools.wraps(func)
    def wrapper_memory(*args, **kwargs):
        mem_used_bef, _ = (tracemalloc.get_traced_memory()) / 10 ** 6         
        value = func(*args, **kwargs)
        mem_used_aft, _ = (tracemalloc.get_traced_memory()) / 10 ** 6      
        mem_incremented = mem_used_aft - mem_used_bef
        profiler_logger.info(f"Memory occupied before {func.__name__!r} was {mem_used_bef} MB")
        profiler_logger.info(f"Memory occupied after {func.__name__!r} was {mem_used_aft} MB")
        profiler_logger.info(f"{func.__name__!r} incremented the memory usage by {mem_incremented} MB")
        return value
    return wrapper_memory