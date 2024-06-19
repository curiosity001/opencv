import time
import Jetson.GPIO as GPIO

# 设置GPIO引脚编号模式为BOARD模式
GPIO.setmode(GPIO.BOARD)

# 定义蜂鸣器的引脚号
buzzer_pin = 24

# 设置蜂鸣器引脚为输出模式
GPIO.setup(buzzer_pin, GPIO.OUT)

# 持续发出报警声
while True:
    # 打开蜂鸣器，发出报警声
    GPIO.output(buzzer_pin, GPIO.LOW)

    # 等待一段时间
    time.sleep(1)

    # 关闭蜂鸣器，停止报警声
    GPIO.output(buzzer_pin, GPIO.HIGH)

    # 等待一段时间
    time.sleep(1)
