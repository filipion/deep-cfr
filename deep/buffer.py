from random import randint

class ReplayBuffer():

    def __init__(self, size):
        self.size = size
        self.stream_idx = 0
        self.data = [-1 for _ in range(self.size)]

    def reset(self):
        self.stream_idx = 0
        self.data = [-1 for _ in range(self.size)]

    def insert(self, replay):
        if self.stream_idx < self.size:
            self.data[self.stream_idx] = replay
        else: 
            idx = randint(0, self.stream_idx)
            if idx < self.size:
                self.data[idx] = replay
        
        self.stream_idx += 1

            

    