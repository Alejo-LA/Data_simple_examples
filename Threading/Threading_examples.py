import time
from threading import Thread


def thread_sleeper(i):
    print("thread %d sleeps for 5 seconds" % i)
    time.sleep(5)
    print("thread %d woke up" % i)

for i in range(10):
    t = Thread(target=thread_sleeper, args=(i,))
    time.sleep(0.5)
    t.start()