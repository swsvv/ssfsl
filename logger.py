import logging
from abc import ABC, abstractmethod
from collections import defaultdict
from logging import DEBUG, INFO, WARN, ERROR, CRITICAL, getLevelName, Logger as PyLogger
from pathlib import Path
from typing import Any, Sequence


class KVWriter(ABC):
    @abstractmethod
    def write(self, data: dict[str, Any], step: int): ...

    @abstractmethod
    def close(self): ...


class MsgWriter(ABC):
    @abstractmethod
    def write_msg(self, msg: str, level: int = ...): ...

    @abstractmethod
    def close(self): ...


class WhiteListWrapper(KVWriter):
    def __init__(self, writer: KVWriter, whitelist: Sequence[str]) -> None:
        super().__init__()
        self.writer = writer
        self.whitelist = frozenset(whitelist)

    def write(self, data: dict[str, Any], step: int):
        filtered = {k: v for k, v in data.items() if k in self.whitelist}
        self.writer.write(filtered, step)

    def close(self):
        self.writer.close()


class CSVWriter(KVWriter):
    def __init__(
        self, file_path: Path, delimiter: str = ",", quotechar: str = '"'
    ) -> None:
        self.file = file_path.open("w+t")
        self.delimiter = delimiter
        self.quotechar = quotechar
        self.keys: list[str] = []  # List for preserving the order

    def write(self, data: dict[str, Any], step: int):
        extra_keys = data.keys() - self.keys
        if len(extra_keys) > 0:
            self.keys.extend(extra_keys)

            self.file.seek(0)
            lines = self.file.readlines()

            # Write the header
            self.file.seek(0)
            for i, key in enumerate(self.keys):
                if i > 0:
                    self.file.write(self.delimiter)
                self.file.write(key)
            self.file.write("\n")

            # Fix the entries
            for line in lines[1:]:
                self.file.write(line[:-1])
                self.file.write(self.delimiter * len(extra_keys))
                self.file.write("\n")

        for i, key in enumerate(self.keys):
            if i > 0:
                self.file.write(self.delimiter)
            value = data.get(key)
            if isinstance(value, str):
                value = value.replace(self.quotechar, self.quotechar + self.quotechar)
                self.file.write(self.quotechar + value + self.quotechar)
            elif value is not None:
                self.file.write(str(value))
        self.file.write("\n")
        self.file.flush()

    def close(self):
        self.file.close()


class ConsoleWriter(KVWriter, MsgWriter):
    def __init__(self, log_file_path: Path, format: str) -> None:
        super().__init__()
        self._msg_logger = logging.getLogger("ConsoleWriterMsgLogger")
        self._msg_logger.setLevel(DEBUG)
        self._msg_logger.addHandler(logging.FileHandler(log_file_path))
        self._msg_logger.addHandler(logging.StreamHandler())
        formatter = logging.Formatter(format)
        for handler in self._msg_logger.handlers:
            handler.setFormatter(formatter)
            self._msg_logger.addHandler(handler)

        self._kv_logger = logging.getLogger("ConsoleWriterKVLogger")
        self._kv_logger.setLevel(DEBUG)
        kv_console_handler = logging.StreamHandler()
        kv_console_handler.setFormatter(logging.Formatter("%(asctime)s  %(message)s"))
        kv_console_handler.setLevel(DEBUG)
        self._kv_logger.addHandler(kv_console_handler)

    def write_msg(self, msg: str, level: int = INFO):
        return self._msg_logger.log(level=level, msg=msg)

    def write(self, data: dict[str, Any], step: int):
        lines: list[str] = []
        for k, v in data.items():
            if isinstance(v, float):
                lines.append(f"{k}: {v:<8.6g}")
            else:
                lines.append(f"{k}: {v}")
        width = max([len(l) for l in lines])
        self._kv_logger.info(
            f"\tstep: {step}\n"
            + "-" * width
            + "\n"
            + "\n".join(lines)
            + "\n"
            + "-" * width
            + "\n"
        )

    def close(self):
        for h in self._msg_logger.handlers:
            h.close()
        for h in self._kv_logger.handlers:
            h.close()


class Logger:
    def __init__(self, file_path: Path, mode: str) -> None:
        self.kv_loggers: list[KVWriter] = []
        self.msg_loggers: list[MsgWriter] = []
        self.data: dict[str, float] = defaultdict(float)
        self.counts: dict[str, int] = defaultdict(int)

        if mode == "debug":
            pass
        else:
            self.add_default_logger(log_file_path=f"{file_path}/exp_{mode}.log")
            self.add_csv_logger(file_path=f"{file_path}/exp_{mode}.csv")

    def record(self, key: str, value: float):
        self.data[key] = value

    def record_mean(self, key: str, value: float):
        self.data[key] = (self.data[key] * self.counts[key] + value) / (
            self.counts[key] + 1
        )
        self.counts[key] += 1

    def dump(self, step: int = 0):
        for logger in self.kv_loggers:
            logger.write(data=self.data, step=step)

        self.data.clear()
        self.counts.clear()

    def info(self, msg: str):
        self.log(msg, INFO)

    def log(self, message: str, level: int = INFO):
        for l in self.msg_loggers:
            l.write_msg(message, level)

    def add_default_logger(
        self,
        log_file_path: Path | str,
        format: str = "%(asctime)s - %(levelname)s - %(message)s",
        whitelist: Sequence[str] | None = None,
    ):
        log_file_path = Path(log_file_path)
        logger = ConsoleWriter(log_file_path, format)
        self.msg_loggers.append(logger)
        if whitelist:
            logger = WhiteListWrapper(logger, whitelist)
        self.kv_loggers.append(logger)

    def add_csv_logger(self, file_path: Path | str):
        file_path = Path(file_path)
        logger = CSVWriter(file_path)
        self.kv_loggers.append(logger)

    def add_kv_logger(self, logger: KVWriter):
        self.kv_loggers.append(logger)


if __name__ == "__main__":
    logger = Logger()
    logger.add_default_logger(Path("test.log"))

    logger.log("Test!", INFO)
