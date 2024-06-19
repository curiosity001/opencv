import threading
import time

class MyThread(threading.Thread):
    def __init__(self):
        super().__init__()
        self.is_running = True

    def run(self):
        while self.is_running:
            print("子线程执行任务")
            time.sleep(1)

    def stop(self):
        self.is_running = False

# 创建并启动子线程
thread = MyThread()
thread.start()

# 主线程等待一定时间后关闭子线程
time.sleep(5)  # 等待5秒
print("主线程关闭子线程")
thread.stop()
thread.join()  # 等待子线程结束

print("主线程退出")

thread.start()
