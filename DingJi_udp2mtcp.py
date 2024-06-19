import threading
import time
from time import sleep
import socket
from queue import Queue

import modbus_tk
import modbus_tk.defines as cst
import modbus_tk.modbus_tcp as modbus_tcp

import random
import requests
import json
import datetime
master1 = modbus_tcp.TcpMaster("192.168.65.204", 8234)
master2 = modbus_tcp.TcpMaster("192.168.65.206", 8234)

# master1 = modbus_tcp.TcpMaster("192.168.31.49", 502)
# master2 = modbus_tcp.TcpMaster("192.168.31.70", 8234)

udp_que1 = Queue()
udp_que2 = Queue()





class udp_mr_server(threading.Thread):
    def __init__(self, threadID, udp_addr, udp_que1, udp_que2):
        threading.Thread.__init__(self)
        self.threadID = threadID
        # UDP的初始化，包括IP地址、端口号设置
        self.udp_addr = udp_addr  # UDP的地址和端口
        # 创建一个UDP socket
        self.udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # 绑定到指定的地址和端口
        self.udp_socket.bind(udp_addr)
        self.udp_que1 = udp_que1
        self.udp_que2 = udp_que2

    def run(self):
        # while (self.cap.isOpened()):
        while True:
            recv_data = self.udp_socket.recvfrom(1024)
            if int(chr(recv_data[0][1])) == 1:
                self.udp_que1.put(recv_data[0])
            if int(chr(recv_data[0][1])) == 2:
                self.udp_que2.put(recv_data[0])

            if self.udp_que1.qsize() >= 50:  # 最多容纳50条待发送的信息
                self.udp_que1.get()
            if self.udp_que2.qsize() >= 50:  # 最多容纳50条待发送的信息
                self.udp_que2.get()


class udp_mtcp_client(threading.Thread):
    def __init__(self, threadID, udp_que, master,ip,port):
        threading.Thread.__init__(self)
        self.threadID = threadID
        # Modbus_tcp的初始化，包括IP地址等
        self.udp_que = udp_que
        self.master = master
        self.master.set_timeout(5.0)

        self.ip=ip
        self.port=port

    def run(self):
        while True:
            udp_data = self.udp_que.get()
            print("data:",udp_data)
            # print("lock:",udp_data.decode('ascii')[1],"  ",udp_data.decode('ascii')[4:6])

            # print(self.data_dict)

            register_number = int(udp_data[4:6].decode('ascii'))
            # print(int(chr(udp_data[-1])))
            try:
                self.master.execute(1, cst.WRITE_MULTIPLE_REGISTERS, register_number, quantity_of_x=1,
                                    output_value=[int(chr(udp_data[-1]))])
            except:
                self.master=modbus_tcp.TcpMaster(self.ip, self.port)
                time.sleep(1)
                print("Data Error",udp_data)


if __name__ == '__main__':
    thread_udp_server = udp_mr_server(1, ('192.168.65.201', 9989), udp_que1, udp_que2)
    thread_mtcp_client1 = udp_mtcp_client(2, udp_que1, master1,"192.168.65.204", 8234)
    thread_mtcp_client2 = udp_mtcp_client(3, udp_que2, master2,"192.168.65.206", 8234)
    # thread_mtcp_client3 = udp_mtcp_client(4, udp_que3, master3,"192.168.16.41", 8234)

    thread_udp_server.start()
    thread_mtcp_client1.start()
    thread_mtcp_client2.start()
    # thread_mtcp_client3.start()

    thread_udp_server.join()
    thread_mtcp_client1.join()
    thread_mtcp_client2.join()
    # thread_mtcp_client3.join()
