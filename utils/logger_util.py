import logging
from logging import Logger
from logging.handlers import TimedRotatingFileHandler
from tensorboardX import SummaryWriter


class MyLogger(object):
    def __init__(self, logger_name, logger_file, summary_dir):
        self.logger = self.init_logger(logger_name, logger_file)
        self.writer = SummaryWriter(summary_dir)

    def info(self, *args, **kwargs):
        return self.logger.info(*args, **kwargs)

    def error(self, *args, **kwargs):
        return self.logger.error(*args, **kwargs)

    def add_scalar(self, *args, **kwargs):
        return self.writer.add_scalar(*args, **kwargs)

    def set_level(self, *args, **kwargs):
        return self.logger.setLevel(*args, **kwargs)

    @staticmethod
    def init_logger(logger_name, log_file):
        if logger_name not in Logger.manager.loggerDict:
            datefmt = "%Y-%m-%d %H:%M:%S"
            # format_str = '[%(asctime)s] [%(levelname)s] [%(filename)s(%(lineno)s line)]: %(message)s'
            format_str = '[%(asctime)s] [%(levelname)s]: %(message)s'
            formatter = logging.Formatter(format_str, datefmt)

            handler = TimedRotatingFileHandler(filename=log_file, when="D", backupCount=3)
            handler.setFormatter(formatter)
            handler.setLevel(logging.INFO)

            console = logging.StreamHandler()
            console.setFormatter(formatter)
            console.setLevel(logging.INFO)

            logger = logging.getLogger(logger_name)
            logger.setLevel(logging.INFO)
            logger.addHandler(handler)
            logger.addHandler(console)
        else:
            logger = logging.getLogger(logger_name)
        return logger


if __name__ == '__main__':
    from config import task_name, log_file, summary_dir

    logger_ = MyLogger(task_name, log_file, summary_dir)
