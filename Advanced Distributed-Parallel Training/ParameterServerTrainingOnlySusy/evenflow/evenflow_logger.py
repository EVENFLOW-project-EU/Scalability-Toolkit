import logging
import sys


class EvenflowLogger:
    def __init__(self, name: str, **kwargs):
        self.name: str = name

        self.log_root: str = kwargs["log_root"]
        self.log_level: int = logging.getLevelName(kwargs.get("log_level", "DEBUG"))

        self._local_logger: logging.Logger = self._create_local_logger()

        self.debug(f"Logger init, using log root {self.log_root}")

    def log(self, msg: str, level: int = logging.DEBUG) -> None:
        """Used for logging local messages."""
        msg_to_be_logged: str = msg.replace("\n", " ")
        self._local_logger.log(level=level, msg=msg_to_be_logged)

    def info(self, msg: str) -> None:
        self.log(msg, logging.INFO)

    def debug(self, msg: str) -> None:
        self.log(msg, logging.DEBUG)

    def error(self, msg: str) -> None:
        self.log(msg, logging.ERROR)

    def _create_local_logger(self) -> logging.Logger:
        """Logger exclusively for this node, used both stdout and a file."""
        new_logger: logging.Logger = logging.getLogger(name=self.name)
        new_logger.propagate = False
        new_logger.setLevel(self.log_level)

        old_factory = logging.getLogRecordFactory()

        def record_factory(*args, **kwargs):
            record = old_factory(*args, **kwargs)
            record.local_ctx = f"{self.name}"
            return record

        logging.setLogRecordFactory(record_factory)

        file_handler = logging.FileHandler(f"{self.log_root}/{self.name}.log", mode="w")
        file_handler.setLevel(self.log_level)
        file_handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s,%(msecs)d %(local_ctx)s %(levelname)s %(message)s",
                datefmt="%H:%M:%S",
            )
        )
        new_logger.addHandler(file_handler)

        stdout_handler: logging.StreamHandler = logging.StreamHandler(sys.stdout)
        stdout_handler.setLevel(self.log_level)
        stdout_handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s,%(msecs)d %(local_ctx)s %(levelname)s %(message)s",
                datefmt="%H:%M:%S",
            )
        )
        new_logger.addHandler(stdout_handler)
        return new_logger
