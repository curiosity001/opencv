import threading


from time import sleep
import socket

udp_addr = ('127.0.0.1', 9999)
udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

def main():
    # udp 通信地址，IP+端口号
    udp_addr = ('127.0.0.1', 9999)
    udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    # 发送数据到指定的ip和端口,每隔1s发送一次，发送10次
    for i in range(10):
        udp_socket.sendto(("Hello,I am a UDP socket for: " + str(i)) .encode('utf-8'), udp_addr)
        print("send %d message" % i)
        sleep(1)

    # 5. 关闭套接字
    udp_socket.close()


class udp_client_thread(threading.Thread):
    def __init__(self, threadID,udp_socket):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.udp_socket=udp_socket
    def run(self):
        # while (self.cap.isOpened()):
        while True:
            self.udp_socket.sendto(("udp test"+str(self.threadID)).encode('utf-8'),udp_addr)
            sleep(1)
        # self.udp_socket.close()



if __name__ == '__main__':
    thread1=udp_client_thread(1,udp_socket)
    thread2 = udp_client_thread(2, udp_socket)
    thread3 = udp_client_thread(3, udp_socket)
    thread4 = udp_client_thread(4, udp_socket)
    thread5 = udp_client_thread(5, udp_socket)
    thread6 = udp_client_thread(6, udp_socket)
    thread7 = udp_client_thread(7, udp_socket)

    thread1.start()
    thread2.start()
    thread3.start()
    thread4.start()
    thread5.start()
    thread6.start()
    thread7.start()

    thread1.join()
    thread2.join()
    thread3.join()
    thread4.join()
    thread5.join()
    thread6.join()
    thread7.join()

