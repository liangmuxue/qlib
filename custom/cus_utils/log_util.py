from loguru import logger as lg
import os
import sys

BASE_LOG_DIR = "custom/data/logs"

def singleton(cls):
    _instance = {}

    def inner():
        if cls not in _instance:
            _instance[cls] = cls()
        return _instance[cls]
    return inner

@singleton
class AppLogger:
    def __init__(self):
        self.app_logger = lg
        filename = "app.log"
        self.set_logger(filename)

    def set_logger(self, filename, filter_type=None, level='DEBUG'):

        # dic = dict(
        #     sink=self.get_log_path(filename),
        #     rotation='500 MB',
        #     retention='30 days',
        #     format="{time}|{level}|{message}",
        #     encoding='utf-8',
        #     level=level,
        #     enqueue=True,
        # )
        # if filter_type:
        #     dic["filter"] = lambda x: filter_type in str(x['level']).upper()
        # self.app_logger.add(**dic)
        config = {
            "handlers": [
                {"sink": sys.stdout, "format": "{time:YYYY-MM-DD hh:mm:ss} - {level} - {message}"},
            ],
            "extra": {"user": "someone"}
        }
        self.app_logger.configure(**config)
        self.app_logger.add(filename, level="DEBUG", rotation="100 MB")
        
        return self.app_logger

    @property
    def get_logger(self):
        return self.app_logger
    
    @staticmethod
    def get_log_path(filename):
        log_path = os.path.join(BASE_LOG_DIR, filename)
        return log_path

    def trace(self, msg):
        self.app_logger.trace(msg)

    def debug(self, msg):
        self.app_logger.debug(msg)

    def info(self, msg):
        self.app_logger.info(msg)

    def success(self, msg):
        self.app_logger.success(msg)

    def warning(self, msg):
        self.app_logger.warning(msg)

    def error(self, msg):
        self.app_logger.error(msg)

    def critical(self, msg):
        self.app_logger.critical(msg)




logger = AppLogger()
logger.set_logger('error.log', filter_type='ERROR')
logger.set_logger('service.log', filter_type='INFO', level='INFO')