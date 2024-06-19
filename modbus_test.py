import modbus_tk.modbus_tcp as mt
import modbus_tk.defines as md
master=mt.TcpMaster("192.168.31.99",8234)
#设置响应等待时间
master.set_timeout(5.0)
#参数说明
#slave 从机地址
#function_code  功能码，在这里我们只使用04（读）和16（写），modbus_tk已经用宏定义的方式实现
#starting_address  读写的开始地址，都是0——31.
#quantity_of_x   连续读写的位数
#output_value   写的值，以迭代的形式，读的时候不是

aa=master.execute(slave=1,function_code=md.READ_INPUT_REGISTERS,starting_address=0,
                  quantity_of_x=32, output_value=1)
print(aa)
aa=master.execute(slave=1,function_code=md.WRITE_MULTIPLE_REGISTERS,starting_address=0,
                  quantity_of_x=1, output_value=[0])