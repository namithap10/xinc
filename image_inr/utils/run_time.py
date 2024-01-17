import time 


class RunTime():
    def __init__(self):
        self.run_time = []
    
    def start(self):
        self.start = time.time()

    def end(self):
        self.end = time.time()
        self.run_time.append(self.end - self.start)

