import cv2
import numpy as np
import datetime
import re
import os

# 定义全局变量
points = []  # 用于存储选取的坐标点
current_point = []  # 当前正在选取的坐标点
regions = {}  # 存储选定的区域坐标点

def mouse_callback(event, x, y, flags, param):
    global current_point

    if event == cv2.EVENT_LBUTTONDOWN:
        current_point = (x, y)
        points.append(current_point)

        if len(points) >= 2:
            cv2.line(frame, points[-2], points[-1], (0, 255, 0), 2)

        cv2.imshow("Camera", frame)

def main():
    # 摄像头URL
    camera_url = 'rtsp://admin:Aust12345@192.168.31.62:554/live'

    cap = cv2.VideoCapture(camera_url)
    if not cap.isOpened():
        print("无法打开摄像头！")
        return

    cv2.namedWindow("Camera")
    cv2.setMouseCallback("Camera", mouse_callback)

    global frame
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        for p in points:
            cv2.circle(frame, p, 5, (0, 0, 255), -1)

        cv2.imshow("Camera", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("c"):
            if len(points) >= 3:
                region_name = input("请输入区域名称: ")
                regions[region_name] = points.copy()
                points.clear()
                frame = np.copy(frame)  # 清除选取的点和连线

    cap.release()
    cv2.destroyAllWindows()

    # 获取当前日期
    current_date = datetime.datetime.now().strftime("%Y-%m-%d")

    # 处理摄像头URL，提取IP地址
    camera_ip = re.search(r'(\d+\.\d+\.\d+\.\d+)', camera_url).group(1)

    # 创建文件夹（如果不存在）
    folder_name = camera_ip
    os.makedirs(folder_name, exist_ok=True)

    # 生成txt文件
    file_name = os.path.join(folder_name, f"{current_date}.txt")
    file_name = re.sub(r'[:]', '_', file_name)  # 将冒号替换成下划线，确保文件名有效
    with open(file_name, "w") as file:
        for region_name, region_points in regions.items():
            points_str = ', '.join([str(point) for point in region_points])
            file.write(f"{region_name}=[{points_str}]\n")

if __name__ == "__main__":
    main()
