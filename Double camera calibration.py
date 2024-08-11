import time
import cv2
import numpy as np
import matplotlib.pylab as plt
import os
                                                    #该模块为相机标定模块
                                                    #该模块标定实现代码为他人模板

print("当前工作目录:", os.getcwd())
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

class Calibrator:
    def __init__(self):
        self.img_size = None  # 图像尺寸
        self.points_world_xyz = []  # 世界坐标
        self.points_pixel_xy = []  # 像素坐标
        self.error = None  # 重投影误差
        self.mtx = None  # 内参矩阵
        self.dist = None  # 畸变系数
        self.rvecs = None  # 旋转矩阵
        self.tvecs = None  # 平移矩阵
        self.calib_info = {}

    def detect(self, cols, rows, folder, show):
        assert ((cols > 0) & (rows > 0))
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        # 标定的文件
        calib_files = [os.path.join(folder, file) for file in os.listdir(folder) if file.endswith('.jpg')]
        calib_files.sort()  # 内部进行排序

        for filename in calib_files:
            img = self.imread(filename)
            if img is None:
                raise FileNotFoundError(f"{filename} 文件没有找到。")
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
            if self.img_size is None:
                self.img_size = gray.shape[::-1]
            else:
                assert gray.shape[::-1] == self.img_size, "所有图像的尺寸必须相同。"

            # 角点粗检测
            ret, corners = cv2.findChessboardCorners(gray, (cols, rows), None)
            if ret:
                # 角点精检测
                corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                # 为每个棋盘格图像创建世界坐标
                point_world_xyz = np.zeros((rows * cols, 3), np.float32)
                point_world_xyz[:, :2] = np.mgrid[0:rows, 0:cols].T.reshape(-1, 2)
                # 添加角点像素坐标，世界坐标
                self.points_pixel_xy.append(corners)
                self.points_world_xyz.append(point_world_xyz)
            else:
                print(f"未检测到角点 {filename}")

            if show:
                img = cv2.drawChessboardCorners(img, (cols, rows), corners, ret)
                title = os.path.basename(filename)
                cv2.imshow(title, img)
                cv2.moveWindow(title, 500, 200)
                cv2.waitKey(1)  # 等待1ms，以便更新窗口

        cv2.destroyAllWindows()  # 关闭所有窗口

    def calib(self):
        # 标定
        self.error, self.mtx, self.dist, self.rvecs, self.tvecs = cv2.calibrateCamera(
            self.points_world_xyz,  # 世界坐标
            self.points_pixel_xy,  # 像素坐标
            self.img_size,  # 图像尺寸
            None, None
        )

    def rectify(self, img):
        return cv2.undistort(img, self.mtx, self.dist)

    def save_calib_info(self, save_file):
        print("重投影误差\n", self.error, "\n")
        print("相机内参\n", self.mtx, "\n")
        print("相机畸变\n", self.dist, "\n")
        print("旋转矩阵\n", self.rvecs, "\n")
        print("平移矩阵\n", self.tvecs, "\n")
        self.calib_info["Error"] = self.error
        self.calib_info["Camera Matrix"] = self.mtx
        self.calib_info["Distortion Coefficients"] = self.dist
        self.calib_info["Rotation Vectors"] = self.rvecs
        self.calib_info["Translation Vectors"] = self.tvecs
        np.save(save_file, self.calib_info)
        print(f"保存标定信息到文件 {save_file}")

    @staticmethod
    def imread(filename: str):
        try:
            img = cv2.imread(filename)
            if img is None:
                raise ValueError(f"无法读取图像 {filename}")
            return img
        except Exception as e:
            print(f"读取图像 {filename} 时发生错误: {e}")
            return None

if __name__ == '__main__':
    # 摄像头索引
    cap_left = cv2.VideoCapture(1)
    cap_right = cv2.VideoCapture(2)

    # 检查摄像头是否成功打开
    if not cap_left.isOpened() or not cap_right.isOpened():
        print("无法打开摄像头")
        exit()

    # 设置保存图像的文件夹路径
    save_folder_left = r"D:\project\biye peoject\imgs_left"
    save_folder_right = r"D:\project\biye peoject\imgs_right"
    if not os.path.exists(save_folder_left):
        os.makedirs(save_folder_left)
    if not os.path.exists(save_folder_right):
        os.makedirs(save_folder_right)

    # 设置循环次数
    num_captures = 5

    # 循环次数计数器
    capture_count = 0
    time_interval = 3

    while capture_count < num_captures:
        # 等待指定的时间间隔
        time.sleep(time_interval)

        # 读取当前帧
        ret_left, frame_left = cap_left.read()
        ret_right, frame_right = cap_right.read()
        if not ret_left or not ret_right:
            print("无法从摄像头读取图像")
            break

        # 构建图像保存路径
        save_path_left = os.path.join(save_folder_left, f"frame_{time.strftime('%Y%m%d_%H%M%S')}.jpg")
        save_path_right = os.path.join(save_folder_right, f"frame_{time.strftime('%Y%m%d_%H%M%S')}.jpg")

        # 保存图像
        cv2.imwrite(save_path_left, frame_left)
        cv2.imwrite(save_path_right, frame_right)
        print(f"图像已保存到 {save_path_left} 和 {save_path_right}")

        # 更新循环次数计数器
        capture_count += 1

    # 释放摄像头资源和关闭窗口
    cap_left.release()
    cap_right.release()
    cv2.destroyAllWindows()

    # 确认图像是否保存成功，并打印捕获结束信息
    print("图像捕获结束，共捕获了 {} 张图像。".format(capture_count))

    # 双目相机标定
    folder_left = r"D:\project\biye peoject\imgs_left"
    folder_right = r"D:\project\biye peoject\imgs_right"
    save_file_left = r"D:\project\biye peoject\calib_info_left.npy"
    save_file_right = r"D:\project\biye peoject\calib_info_right.npy"
    show = True  # 根据需要设置是否显示结果
    cols = 8  # 棋盘格行角点
    rows = 6  # 棋盘格列角点

    calibrator_left = Calibrator()
    calibrator_right = Calibrator()

    calibrator_left.detect(cols, rows, folder_left, show)
    calibrator_right.detect(cols, rows, folder_right, show)

    calibrator_left.calib()
    calibrator_right.calib()

    calibrator_left.save_calib_info(save_file_left)
    calibrator_right.save_calib_info(save_file_right)
