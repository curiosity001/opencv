# import modbus_tk.modbus_tcp as mt
# import modbus_tk.defines as md
#
# sign=[0,0,0]
#
# master=mt.TcpMaster("192.168.31.70",8234) #向某个地址发送数据
# #设置响应等待时间
# master.set_timeout(5.0)
#
#
# master.execute(slave=1, function_code=md.WRITE_MULTIPLE_REGISTERS, starting_address=9,
#                                      quantity_of_x=3, output_value=sign)



import threading
import time
from time import sleep
import socket
from queue import Queue

# 导入modbus_tk模块
import modbus_tk
# 导入modbus_tk库中的常量定义
import modbus_tk.defines as cst
# 导入modbus_tk库中用于处理Modbus TCP协议的部分
import modbus_tk.modbus_tcp as modbus_tcp

import random

# master1 = modbus_tcp.TcpMaster("127.0.0.1", 502)
master2 = modbus_tcp.TcpMaster("192.168.31.70", 8234)


udp_que1 = Queue()
udp_que2 = Queue()

#人员检测接收udp线程
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
            # print(type(recv_data[0][1]))  # <class 'int'>
            # print(recv_data[0][1])
            if int(chr(recv_data[0][1])) == 1:
                self.udp_que1.put(recv_data[0])
            if int(chr(recv_data[0][1])) == 2:
                self.udp_que2.put(recv_data[0])
            if self.udp_que1.qsize() >= 50:  # 最多容纳50条待发送的信息
                self.udp_que1.get()
            if self.udp_que2.qsize() >= 50:  # 最多容纳50条待发送的信息
                self.udp_que2.get()
            # print(self.udp_que_1.put(recv_data[0]))
        # self.udp_socket.close()


#疲劳检测接收udp线程
class udp_mr_server_face(threading.Thread):
    def __init__(self, threadID, udp_addr, udp_que1):
        threading.Thread.__init__(self)
        self.threadID = threadID
        # UDP的初始化，包括IP地址、端口号设置
        self.udp_addr = udp_addr  # UDP的地址和端口
        # 创建一个UDP socket
        self.udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # 绑定到指定的地址和端口
        self.udp_socket.bind(udp_addr)
        self.udp_que1 = udp_que1
        # self.udp_que2 = udp_que2

    def run(self):
        # while (self.cap.isOpened()):
        while True:
            recv_data = self.udp_socket.recvfrom(1024)
            # print(type(recv_data[0][1]))  # <class 'int'>
            # print(recv_data)
            print("%s" % recv_data[0].decode("utf-8"))
            self.udp_que1.put(recv_data[0])
            if self.udp_que1.qsize() >= 50:  # 最多容纳50条待发送的信息
                self.udp_que1.get()
            # print(self.udp_que_1.put(recv_data[0]))
        # self.udp_socket.close()


class udp_mtcp_client(threading.Thread):
    def __init__(self, threadID, udp_que, master):
        threading.Thread.__init__(self)
        self.threadID = threadID
        # Modbus_tcp的初始化，包括IP地址等
        self.udp_que = udp_que
        self.master = master
        self.master.set_timeout(5.0)

    def run(self):
        while True:
            self.master.execute(1, cst.WRITE_MULTIPLE_REGISTERS, 5, quantity_of_x=1,
                                    output_value=1)



if __name__ == '__main__':
    # thread_udp_server = udp_mr_server_face(1, ('192.168.31.86', 9999), udp_que1)
    # thread_udp_server = udp_mr_server(1, ('192.168.31.86', 9999), udp_que1,udp_que2)
    thread_mtcp_client1 = udp_mtcp_client(2, udp_que2, master2)
    # thread_mtcp_client2 = udp_mtcp_client(3, udp_que2, master1)

    # thread_udp_server.start()
    thread_mtcp_client1.start()
    # thread_mtcp_client2.start()

    # thread_udp_server.join()
    thread_mtcp_client1.join()
    # thread_mtcp_client2.join()
