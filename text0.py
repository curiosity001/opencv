import cv2
import numpy as np

# 存储用户点击的坐标点
points = []

# 读取图片
image = cv2.imread('1.jpg')

# 复制一份图片用于显示选取区域
image_copy = image.copy()

# 定义鼠标事件回调函数
def get_coordinates(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        print(f"坐标点：({x}, {y})")
        if len(points) > 1:
            cv2.line(image_copy, points[-2], points[-1], (255, 0, 0), 2)
            cv2.imshow('image', image_copy)

# 创建窗口并显示图片
cv2.imshow('image', image_copy)
cv2.setMouseCallback('image', get_coordinates)

# 定义键盘按键事件处理函数
def on_key_press(key):
    if key == ord('f'):  # 按下 'f' 键表示完成绘制并退出程序
        finish_drawing_and_exit()
    elif key == ord('r'):  # 按下 'r' 键表示重新绘制
        redraw()

# 定义完成绘制函数
def finish_drawing_and_exit():
    global points
    print("绘制完成！")
    points_array = np.array(points)  # 将坐标点存储为 NumPy 数组
    print("坐标矩阵：", points_array)
    # 在这里保存坐标到文件
    np.savetxt('points.txt', points_array.reshape(1, -1), fmt='%d', delimiter=',')  # 保存坐标为一行的文本文件
    cv2.destroyAllWindows()  # 关闭窗口
    exit(0)  # 退出程序

# 定义重新绘制函数
def redraw():
    global points, image_copy
    print("重新绘制！")
    points = []
    image_copy = image.copy()
    cv2.imshow('image', image_copy)

# 等待用户手动获取坐标和按键事件
while True:
    key = cv2.waitKey(0) & 0xFF
    if key == 27:  # 按下 'ESC' 键退出程序
        cv2.destroyAllWindows()
        exit(0)
    on_key_press(key)

# 闭合首尾点
if len(points) > 1:
    cv2.line(image_copy, points[0], points[-1], (255, 0, 0), 2)
    cv2.imshow('image', image_copy)

# 等待用户按任意键关闭窗口
cv2.waitKey(0)
cv2.destroyAllWindows()