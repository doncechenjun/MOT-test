import cv2
import numpy as np

# 读取视频文件
cap = cv2.VideoCapture('your_video.mp4')

# 创建BackgroundSubtractor对象，用于背景减除
fgbg = cv2.createBackgroundSubtractorMOG2()

# 创建空白图像用于绘制轨迹
trajectory_image = np.zeros((720, 1280, 3), dtype=np.uint8)

# 记录上一帧中心点的位置
prev_center = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 对当前帧应用背景减除
    fgmask = fgbg.apply(frame)

    # 对结果进行形态学操作，去除噪声
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)

    # 查找轮廓
    contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 计算中心点并绘制轨迹
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 50:  # 设置一个阈值来过滤掉过小的轮廓
            M = cv2.moments(contour)
            if M["m00"] != 0:
                center_x = int(M["m10"] / M["m00"])
                center_y = int(M["m01"] / M["m00"])
                current_center = (center_x, center_y)
                if prev_center is not None:
                    cv2.line(trajectory_image, prev_center, current_center, (0, 255, 0), 2)
                prev_center = current_center

    # 显示结果
    cv2.imshow('frame', frame)
    cv2.imshow('trajectory', trajectory_image)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
