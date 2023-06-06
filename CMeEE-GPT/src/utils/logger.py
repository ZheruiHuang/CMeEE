# txt logger

import logging
import logging.handlers
import os
import sys


class TxtLogger(object):
    def __init__(self, name, log_dir, level=logging.DEBUG):
        self.name = name
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        self.level = level
        self.logger = self._get_logger()

    def _get_logger(self):
        logger = logging.getLogger(self.name)
        logger.setLevel(self.level)
        formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
        handler = logging.handlers.TimedRotatingFileHandler(
            os.path.join(self.log_dir, self.name + ".log"), when="D", interval=1, backupCount=7, encoding='utf-8',
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger

    def info(self, msg):
        self.logger.info(msg)

    def debug(self, msg):
        self.logger.debug(msg)

    def warning(self, msg):
        self.logger.warning(msg)


if __name__ == "__main__":
    logger = TxtLogger("test", "./logs")
    logger.info("test")
    logger.debug("test")
    logger.warning("test")
