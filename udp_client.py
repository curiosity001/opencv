#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
udp通信例程：udp client端，修改udp_addr元组里面的ip地址，即可实现与目标机器的通信，
此处以单机通信示例，ip为127.0.0.1，实际多机通信，此处应设置为目标服务端ip地址
"""

__author__ = "River.Yang"
__date__ = "2021/4/30"
__version__ = "1.0.0"

#这里导入了sleep函数和socket模块，sleep函数用于在发送数据包之间进行延时，socket模块提供了网络通信的功能
from time import sleep
import socket

def main():
    # udp 通信地址，IP+端口号
    udp_addr = ('127.0.0.1', 9999)  #这里定义了一个UDP通信地址，其中127.0.0.1表示本地主机，9999是指定的端口号。
    udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)  #这里使用socket.socket函数创建了一个UDP套接字，socket.AF_INET表示使用IPv4地址，socket.SOCK_DGRAM表示使用UDP协议。

    # 发送数据到指定的ip和端口,每隔1s发送一次，发送10次
    #通过udp_socket.sendto函数将数据发送到指定的IP地址和端口号，其中("Hello, I am a UDP socket for: " + str(i))是要发送的消息，.encode('utf-8')将消息转换为字节流发送。
    # 循环会执行10次，每次发送一条消息，并在发送后打印消息发送的序号。sleep(1)函数使程序延时1秒，以便在发送每条消息之间有间隔。
    for i in range(10):
        udp_socket.sendto(("Hello,I am a UDP socket for: " + str(i)) .encode('utf-8'), udp_addr)
        print("send %d message" % i)
        sleep(1)

    # 5. 关闭套接字
    udp_socket.close()

#这里使用if __name__ == '__main__':来判断当前模块是否作为主程序运行。如果是，则打印当前版本和提示信息，并调用main函数执行主程序逻辑。
if __name__ == '__main__':
    print("当前版本： ", __version__)
    print("udp client ")
    main()

