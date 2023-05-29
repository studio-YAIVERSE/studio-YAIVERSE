import os
from contextlib import contextmanager


@contextmanager
def at_working_directory(work_dir):
    prev = os.getcwd()
    try:
        os.chdir(work_dir)
        yield
    finally:
        os.chdir(prev)
