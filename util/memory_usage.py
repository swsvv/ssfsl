import os
import psutil


def print_memory_usage() -> str:
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    memory_usage = memory_info.rss  # in bytes
    msg = f"Memory usage: {memory_usage / (1024 ** 2):.2f} MB"
    return msg
