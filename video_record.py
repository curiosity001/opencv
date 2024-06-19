import cv2
import datetime
import threading
import time
def video_rec(url,video_number):
    cap=cv2.VideoCapture(url)
    width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps=25.0
    video_length=30*60
    filename_prefix = "D:\C--transfer\桌面\gb_video4/"
    frame_count=0


    while True:
        if frame_count==0:
            filename=filename_prefix+str(video_number)+".mp4"
            out=cv2.VideoWriter(filename,fourcc,fps,(width,height))#这行代码创建了一个cv2.VideoWriter对象，用于将视频帧写入视
            # 频文件中。它接受以下参数:filename: 视频文件的名称;fourcc: 视频编码器，这里使用了MP4视频编码器;fps: 视频的帧率;(width, height): 视频帧的宽度和高度
            print("start recording"+filename)

        ret,frame=cap.read()
        if frame_count>=fps*video_length:
            out.release()
            print("Finish"+filename)
            break

        else:
            out.write(frame)
            frame_count+=1

if __name__ == '__main__':
    # video_rec('rtsp://admin:Aust12345@192.168.31.68:554/live',111)

    threading.Thread(target=video_rec,args=('rtsp://admin:cs123456@192.168.12.133:554/live',1),daemon=True).start()
    threading.Thread(target=video_rec,args=('rtsp://admin:cs123456@192.168.12.134:554/live',2),daemon=True).start()
    threading.Thread(target=video_rec,args=('rtsp://admin:cs123456@192.168.12.135:554/live',3),daemon=True).start()
    threading.Thread(target=video_rec,args=('rtsp://admin:cs123456@192.168.12.136:554/live',4),daemon=True).start()
    threading.Thread(target=video_rec,args=('rtsp://admin:cs123456@192.168.12.137:554/live',5),daemon=True).start()
    threading.Thread(target=video_rec,args=('rtsp://admin:cs123456@192.168.12.138:554/live',6),daemon=True).start()
    threading.Thread(target=video_rec,args=('rtsp://admin:cs123456@192.168.12.139:554/live',7),daemon=True).start()
    threading.Thread(target=video_rec,args=('rtsp://admin:cs123456@192.168.12.140:554/live',8),daemon=True).start()
    time1=time.time()
    while True:
        time2=time.time()
        print("recording",time2-time1)
        time.sleep(2)
