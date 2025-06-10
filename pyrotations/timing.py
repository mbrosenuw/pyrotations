import time
from contextlib import ContextDecorator


class timing(ContextDecorator):
    def __init__(self, title="Operation"):
        self.title = title

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, *exc):
        self.end_time = time.time()
        self.elapsed_time = self.end_time - self.start_time
        print(f'Time elapsed for {self.title}: {self.elapsed_time:.6f} seconds')

    def get_time(self):
        return getattr(self, 'elapsed_time', None)