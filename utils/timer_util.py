from functools import wraps
import time
import datetime


def ts_print(*args):
    cur_time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
    print("[%s] %s" % (cur_time_str, " ".join([str(arg) for arg in args])))


def timer(func):
    @wraps(func)
    def with_timer(*args, **kwargs):
        tic = time.time()
        res = func(*args, **kwargs)
        toc = time.time()
        ts_print("%s costs: %.2f(s)" % (func.__name__, (toc - tic)))
        return res
    return with_timer


def logger_timer(func, logger):
    @wraps(func)
    def with_timer(*args, **kwargs):
        tic = time.time()
        res = func(*args, **kwargs)
        toc = time.time()
        logger.info("%s costs: %.2f(s)" % (func.__name__, (toc - tic)))
        return res
    return with_timer


GMT_FORMAT = '%a %b %d %Y %H:%M:%S GMT+0800 (China Standard Time)'


def gmt_transpose(gmt_str):
    return datetime.datetime.strptime(gmt_str, GMT_FORMAT).__str__()
