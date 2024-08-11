import cv2
import numpy as np

# 加载标定文件
calib_info_left = np.load('calib_info_left.npy', allow_pickle=True).item()
calib_info_right = np.load('calib_info_right.npy', allow_pickle=True).item()

# 提取内参矩阵和畸变系数
camera_matrix_left = calib_info_left['Camera Matrix']
dist_coeffs_left = calib_info_left['Distortion Coefficients']
camera_matrix_right = calib_info_right['Camera Matrix']
dist_coeffs_right = calib_info_right['Distortion Coefficients']

# 提取旋转向量和平移向量
tvecs_left = calib_info_left['Translation Vectors']
tvecs_right = calib_info_right['Translation Vectors']

# 计算基线距离
tvecs_left_avg = np.mean(tvecs_left, axis=0)
tvecs_right_avg = np.mean(tvecs_right, axis=0)
baseline = np.linalg.norm(tvecs_left_avg - tvecs_right_avg)
print("基线距离（单位：米）:", baseline)

# 计算焦距
focal_length_x = camera_matrix_left[0, 0]
focal_length_y = camera_matrix_left[1, 1]
print("焦距 f_x（单位：像素）:", focal_length_x)
print("焦距 f_y（单位：像素）:", focal_length_y)

# 计算Q矩阵
Q = np.float32([[-focal_length_x / baseline, 0, (camera_matrix_left[0, 2] - camera_matrix_right[0, 2]) / baseline, 0],
                [0, -focal_length_y / baseline, (camera_matrix_left[1, 2] - camera_matrix_right[1, 2]) / baseline, 0],
                [0, 0, -(focal_length_x + focal_length_y) / baseline, focal_length_y],
                [0, 0, 1, 0]])

# 初始化两个摄像头
cap1 = cv2.VideoCapture(0)  # 通常0是默认的摄像头
cap2 = cv2.VideoCapture(1)  # 1通常是第二个摄像头

# 检查摄像头是否成功打开
if not cap1.isOpened() or not cap2.isOpened():
    print("无法打开摄像头")
    exit()

# 循环获取摄像头图像
while True:
    # 读取一帧
    ret, frame1 = cap1.read()
    # 如果正确读取帧，ret为True
    if not ret:
        print("无法从摄像头1读取帧")
        break

    # 读取摄像头2的帧
    ret2, frame2 = cap2.read()
    if not ret2:
        print("无法从摄像头2读取帧")
        break

    # 畸变矫正和重投影
    h, w = frame1.shape[:2]  # 获取图像的高度和宽度
    new_camera_matrix_left = camera_matrix_left.copy()  # 复制内参矩阵
    new_camera_matrix_right = camera_matrix_right.copy()

    # 设置内参矩阵中的主点坐标为图像中心
    new_camera_matrix_left[0, 2] = w / 2
    new_camera_matrix_left[1, 2] = h / 2
    new_camera_matrix_right[0, 2] = w / 2
    new_camera_matrix_right[1, 2] = h / 2

    # 畸变矫正
    undistorted_frame1 = cv2.undistort(frame1, camera_matrix_left, dist_coeffs_left, None, new_camera_matrix_left)
    undistorted_frame2 = cv2.undistort(frame2, camera_matrix_right, dist_coeffs_right, None, new_camera_matrix_right)

    # 灰度化图像
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # 应用高斯模糊
    blurred1 = cv2.GaussianBlur(gray1, (5, 5), 0)
    blurred2 = cv2.GaussianBlur(gray2, (5, 5), 0)

    # 双边滤波
    bilateral1 = cv2.bilateralFilter(blurred1, 9, 75, 75)
    bilateral2 = cv2.bilateralFilter(blurred2, 9, 75, 75)

    # 锐化滤波（使用拉普拉斯算子）
    kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype=np.float32)
    sharpened1 = cv2.filter2D(bilateral1, -1, kernel)
    kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype=np.float32)
    sharpened2 = cv2.filter2D(bilateral2, -1, kernel)

    # 创建CLAHE对象(提升图像对比度)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    # 应用CLAHE
    clahe_image1 = clahe.apply(sharpened1)
    clahe_image2 = clahe.apply(sharpened2)

    # 显示CLAHE处理后的图像
    cv2.imshow('CLAHE Image 1', clahe_image1)
    cv2.imshow('CLAHE Image 2', clahe_image2)

    # 计算视差图
    # 配置StereoBM或StereoSGBM算法的参数
    window_size = 9
    min_disp = 40    #视差的最小值
    num_disp = 64 - min_disp  #视差范围
    stereo = cv2.StereoSGBM_create(minDisparity=min_disp,
                                   numDisparities=num_disp,
                                   blockSize=window_size,
                                   uniquenessRatio=10,
                                   speckleWindowSize=100,
                                   speckleRange=32,
                                   disp12MaxDiff=1)

    # 计算视差
    disparity = stereo.compute(clahe_image1, clahe_image2)

    disparity_normalized = cv2.normalize(disparity, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    cv2.imshow('Disparity Map', disparity_normalized)
    # 相机参数
    focal_length = 0.0025  # 焦距

    # 计算深度图
    points_3D = cv2.reprojectImageTo3D(disparity, Q)
    depth_normalized = cv2.normalize(points_3D, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    cv2.imshow('Depth Map', depth_normalized)

    # 假设已经有了深度图变量 'depth_normalized' 和 'depth'

    # 找到深度图中的最小值及其对应的像素坐标
    min_depth_value = np.min(depth_normalized[depth_normalized > 0])  # 忽略深度为0的点
    min_depth_coords = np.unravel_index(np.argmin(depth_normalized), depth_normalized.shape)

    # 将像素坐标转换为空间坐标
    x = (min_depth_coords[1] - camera_matrix_left[0, 2]) * min_depth_value / focal_length_x
    y = (min_depth_coords[0] - camera_matrix_left[1, 2]) * min_depth_value / focal_length_y
    z = min_depth_value

    # 打印空间坐标
    print("最浅物体的空间坐标（X, Y, Z）:", x, y, z)

    # 显示最浅物体的深度值
    print("最浅物体的深度值（单位：米）:", min_depth_value)

    # 在图像上绘制标记和显示坐标
    # 选择一个图像来显示，这里我们使用 rectified_frame1
    display_image = depth_normalized.copy()

    # 绘制一个圆来标记最浅点
    cv2.circle(display_image, (min_depth_coords[1], min_depth_coords[0]), 5, (0, 255, 0), -1)

    # 将坐标转换为字符串
    coord_text = f"X: {x:.2f}, Y: {y:.2f}, Z: {z:.2f}"

    # 在图像上显示坐标
    cv2.putText(display_image, coord_text, (min_depth_coords[1], min_depth_coords[0] + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # 显示图像
    cv2.imshow('Depth Map with Coordinates', display_image)


    # 按'q'键退出循环
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break



    # 释放摄像头资源
cap1.release()
cap2.release()

# 关闭所有OpenCV窗口
cv2.destroyAllWindows()