import multiprocessing as mp
import multiprocessing.pool
from multiprocessing.pool import ThreadPool

def run_function_different_arguments_parallel(
                                                function, 
                                                arguments, 
                                                parallel = True, 
                                                all_success=True,
                                                signal=None, 
                                                use_thread=False,
                                                *args, **kwargs):
    """
    Call functions in parallel
    :param function: f(argument, **kwargs)
    :param arguments: {i: argument}
    :param all_success: (boolean) the function will raise an exception if one of the runs
        fail and all_success is True
    :param signal: (function) calls this function after generating the jobs. It's used to test
        KeyboardInterrupt, and the signal is a mock of KeyboardInterrupt.
    :param parallel: (boolean) The code is run in parallel only if it's True.
    :param threads: (int) Uses threads instead of processes if threads > 0
    :param args: additional arguments of function
    :param kwargs: additional arguments of function
    :return: {int: output of f(arguments[i])}
    """
    # Maybe later we enable this feature.
    #thread = False
    if not parallel:
        results = {}
        for key, argument in arguments.items():
            _args = (argument, ) + args
            results[key] = function(*_args, **kwargs)
        return results
    else:
        jobs = {}

        n_jobs = min(len(arguments), mp.cpu_count())

        if use_thread:
            threads = len(arguments)
            pool = ThreadPool(threads)
        else:
            pool = mp.Pool(processes=n_jobs)

        try:
            for key, argument in arguments.items():
                job = pool.apply_async(function, args=(argument, ) + args, kwds=kwargs)
                jobs[key] = job
            pool.close()
            pool.join()
            if signal is not None:
                signal(1)
        except KeyboardInterrupt:
            print("Ctrl+c received, terminating and joining pool.")
            pool.terminate()
            pool.join()
            return -1

        results = {}
        n_retry = 5 
        for key in arguments.keys():
            for count in range(n_retry): # retry 5 times before raise error. 
                try:
                    results[key] = jobs[key].get()
                    break 
                except Exception as e:
                    # if all_success:
                    #     raise e
                    if count == n_retry - 1:
                        raise e
                    else:
                        print("job failed")
                        print(argument)
                        print(e)
                        print(args)
                        print(kwargs)
                        print('Retrying ...')
        return results