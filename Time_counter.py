import time


class TimeCounter:
    def __init__(self):
        self.frame_id = -1
        self.time1 = None
        self.last_time = None

    def set_start_time(self, frame_id, print_flag=True):
        self.time1 = time.time()
        self.frame_id = frame_id
        if print_flag:
            print("At frame_%d start time recorded" % self.frame_id)

    def print_passed_time(self):
        self.last_time = time.time()
        passed_time = self.last_time - self.time1
        print("At frame_%d, passed time since start time: %.4f s" % (self.frame_id, passed_time))


    def print_passed_time_since_last(self):
        new_time = time.time()
        passed_time = new_time - self.last_time
        self.last_time = new_time
        print("At frame_%d, passed time since last recording: %.4f s" % (self.frame_id, passed_time))

