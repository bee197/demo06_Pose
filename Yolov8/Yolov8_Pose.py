import collections
import math
from collections import namedtuple, OrderedDict
import time
import cv2
import numpy as np
import tensorrt as trt
import torch
from torchvision import ops

from Yolov8.main import LoadEngineModel
from utils import format_img, gstreamer_pipeline
from utils2 import LoadLSTMEngine


KPS_COLORS = [[0, 255, 0], [0, 255, 0], [0, 255, 0], [0, 255, 0], [0, 255, 0],
              [255, 128, 0], [255, 128, 0], [255, 128, 0], [255, 128, 0],
              [255, 128, 0], [255, 128, 0], [51, 153, 255], [51, 153, 255],
              [51, 153, 255], [51, 153, 255], [51, 153, 255], [51, 153, 255]]

SKELETON = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13],
            [6, 7], [6, 8], [7, 9], [8, 10], [9, 11], [2, 3], [1, 2], [1, 3],
            [2, 4], [3, 5], [4, 6], [5, 7]]

LIMB_COLORS = [[51, 153, 255], [51, 153, 255], [51, 153, 255], [51, 153, 255],
               [255, 51, 255], [255, 51, 255], [255, 51, 255], [255, 128, 0],
               [255, 128, 0], [255, 128, 0], [255, 128, 0], [255, 128, 0],
               [0, 255, 0], [0, 255, 0], [0, 255, 0], [0, 255, 0], [0, 255, 0],
               [0, 255, 0], [0, 255, 0]]
INDEX = [0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
NUM_1 = 0
STACK_SIZE = 5

class LoadPoseEngine:
    def __init__(self, model_path, confidence=0.6, nms_thresh=0.4):
        device = torch.device('cuda:0')  # default
        logger = trt.Logger(trt.Logger.INFO)
        Binding = namedtuple('Binding', ('name', 'dtype', 'shape', 'data', 'ptr'))
        # 加载模型
        self.confidence = confidence
        self.nms_threshold = nms_thresh
        self.input_shape = (1, 3, 640, 640)  # default
        self.N, self.C, self.H, self.W = self.input_shape
        self.bindings = OrderedDict()
        self.bindings_addrs = OrderedDict()
        self.context = None
        self.num_nodes = 17  # 关节点的个数
        self.dif_w, self.dif_h, self.factor = 0, 0, 1
        with open(model_path, 'rb') as f, trt.Runtime(logger) as runtime:
            model = runtime.deserialize_cuda_engine(f.read())
            if model is None:
                print("模型加载失败!!")
            for index in range(model.num_bindings):
                name = model.get_binding_name(index)
                dtype = trt.nptype(model.get_binding_dtype(index))
                shape = model.get_binding_shape(index)
                data = torch.from_numpy(np.empty(shape, dtype=dtype)).to(device)
                self.bindings[name] = Binding(name, dtype, shape, data, int(data.data_ptr()))
            self.bindings_addrs = OrderedDict((n, d.ptr) for n, d in self.bindings.items())
            self.context = model.create_execution_context()

    def _model_process(self, img_src):
        # 获取处理的结果
        img_blob, dif_w, dif_h, factor = format_img(img_src)
        self.dif_w, self.dif_h, self.factor = dif_w, dif_h, factor
        # print('预处理完毕')
        self.bindings_addrs['images'] = img_blob.data_ptr()
        self.context.execute_v2(list(self.bindings_addrs.values()))
        out_prob = self.bindings['output0'].data.squeeze().permute(1, 0)

        # 以下这段在GPU上操作, 并且要用矩阵操作, 可以节约时间, 处理成NMSBOXES函数可接受的数据格式
        # print("out_prob", out_prob.shape)
        flag = out_prob[:, 4] > self.confidence
        out_prob = out_prob[flag]
        # 得到x1, y1
        out_prob[:, 0] -= out_prob[:, 2] / 2 + dif_w
        out_prob[:, 1] -= out_prob[:, 3] / 2 + dif_h
        # 得到x2, y2
        out_prob[:, 2] += out_prob[:, 0]
        out_prob[:, 3] += out_prob[:, 1]
        out_prob[:, :4] *= factor
        # 原3,4为宽高 现得到x2,y2 根据不同的nms算法的输入参数来进行调整
        bboxes = out_prob[:, :4]
        scores = out_prob[:, 4]
        points = out_prob[:, 5:]

        return bboxes, scores, points  # 处理完后把数据做最后的NMS处理

    def _nms(self, bboxes, scores, points):
        indices = ops.nms(bboxes, scores, self.nms_threshold)
        # print('NMS处理完成')
        bboxes = bboxes[indices]
        scores = scores[indices]
        points = points[indices]

        # 处理关节点的正确位置
        for i in range(self.num_nodes):
            points[:, i * 3] = (points[:, i * 3] - self.dif_w) * self.factor
            points[:, i * 3 + 1] = (points[:, i * 3 + 1] - self.dif_h) * self.factor
            points[:, i * 3: i * 3 + 2] = torch.round(points[:, i * 3: i * 3 + 2])
        # ----
        # points shape is [N, 17, 3]
        # N is number of poses, 17 is keypoints, 3 is (x, y, confidence)
        keypoint_conf = torch.zeros((points.shape[0], self.num_nodes))
        for i in range(self.num_nodes):
            keypoint_conf[:, i] = points[:, i * 3 + 2]
        keypoint_conf_sum = keypoint_conf.sum(dim=1)
        # print(keypoint_conf_sum)
        valid_poses_index = np.where(keypoint_conf_sum > 10)[0]
        bboxes = bboxes[valid_poses_index]
        scores = scores[valid_poses_index]
        points = points[valid_poses_index]
        return torch.round(bboxes), scores, points

    def __call__(self, img_src):
        self._model_process(img_src)
        bboxes, scores, points = self._model_process(img_src)
        bboxes, scores, points = self._nms(bboxes, scores, points)
        return bboxes, scores, points

    def draw_pose(self, np_img, bboxes, scores, points):
        for (bbox, score, kpt) in zip(bboxes, scores, points):
            # num = NUM_1
            cv2.rectangle(np_img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 4)
            for i in range(19):
                if i < 17:
                    # NUM_1 += 1
                    px, py, ps = kpt[i * 3: i * 3 + 3]
                    if ps > self.confidence:

                        kcolor = KPS_COLORS[i]
                        px, py = int(round(px)), int(round(py))
                        radius = 3
                        cv2.circle(np_img, (px, py), radius, kcolor, -1)
                        cv2.putText(np_img, str(i), (px, py), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                xi, yi = SKELETON[i]
                pos1_s = kpt[(xi - 1) * 3 + 2]
                pos2_s = kpt[(yi - 1) * 3 + 2]
                if pos1_s > self.confidence and pos2_s > self.confidence:
                    limb_color = LIMB_COLORS[i]
                    pos1_x, pos1_y = int(round(kpt[(xi - 1) * 3])), int(round(kpt[(xi - 1) * 3 + 1]))
                    pos2_x, pos2_y = int(round(kpt[(yi - 1) * 3])), int(round(kpt[(yi - 1) * 3 + 1]))
                    cv2.line(np_img, (pos1_x, pos1_y), (pos2_x, pos2_y),
                             limb_color, 4)

# 设置gstreamer管道参数
def gstreamer_pipeline(
    capture_width=1280, #摄像头预捕获的图像宽度
    capture_height=720, #摄像头预捕获的图像高度
    display_width=1280, #窗口显示的图像宽度
    display_height=720, #窗口显示的图像高度
    framerate=60,       #捕获帧率
    flip_method=0,      #是否旋转图像
):
    return (
        "nvarguscamerasrc ! "
        "video/x-raw(memory:NVMM), "
        "width=(int)%d, height=(int)%d, "
        "format=(string)NV12, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
        % (
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )

def main():
    avg_hip_y_pre = None
    left_angle_pre = None
    right_angle_pre = None
    left_hand_pre = None
    right_hand_pre = None
    max_frame = 1
    frame = 0
    max_people = 3
    punch_num = np.zeros(max_people)
    s = None
    capture_width = 1280
    capture_height = 720
    display_width = 1280
    display_height = 720
    framerate = 60
    flip_method = 0
    try:

        # 创建管道
        print(gstreamer_pipeline(capture_width, capture_height, display_width, display_height, framerate, flip_method))

        # 管道与视频流绑定
        # cap = cv2.VideoCapture(gstreamer_pipeline(flip_method=0), cv2.CAP_GSTREAMER)
        # cap.set(cv2.CAP_PROP_FPS, 30)
        cap = cv2.VideoCapture(0)
        print('摄像头打开成功！')
        # 创建模型
        infer = LoadPoseEngine('yolov8n-pose.engine')  # 加载模型
        # infer = LoadEngineModel()
        print('模型加载完成！')
        # cap = cv2.VideoCapture('image/挥拳5.mp4')

        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (640, 640))

        while 1:
            start = time.time()
            if frame == max_frame:
                frame = 0
            f, np_img = cap.read()
            np_img = cv2.resize(np_img, (540, 360))
            # if not f: break
            # np_img = cv2.imread("image/stand1.jpg")  # 替换为您的图片路径

            # s = time.time()
            bboxes, scores, points = infer(np_img)
            # e = time.time()  # 防止计算的fps分子为0
            scores = scores.cpu().numpy()
            bboxes = bboxes.to(torch.int32).cpu().numpy()
            points = points.cpu().numpy()  # A x 51条数据 17个点 每个点3个信息 分别是该点的x坐标，y坐标，置信度

            # ---------------------------------------------------------#
            #   每n帧计算一次
            # ---------------------------------------------------------#

            if frame == 0:

                # ---------------------------------------------------------#
                #   跌倒决策1
                # ---------------------------------------------------------#

                # e
                e = time.time()
                # now
                left_hip_y = points[:, 11 * 3 + 1]  # 取左臀部y坐标
                right_hip_y = points[:, 12 * 3 + 1]  # 取右臀部y坐标
                avg_hip_y = (left_hip_y + right_hip_y) / 2  # 取平均臀部y坐标
                # 初始化
                if avg_hip_y_pre is None:
                    avg_hip_y_pre = avg_hip_y
                if s is None:
                    s = time.time()
                # v_y 平均髋关节下降速度
                if avg_hip_y.shape[0] == avg_hip_y_pre.shape[0]:
                    v_y = (avg_hip_y - avg_hip_y_pre) / (e - s)
                # pre
                avg_hip_y_pre = avg_hip_y

                w_ = bboxes[:, 2] - bboxes[:, 0]
                h_ = bboxes[:, 3] - bboxes[:, 1]
                if v_y.shape[0] == h_.shape[0]:
                    indices1 = np.where(v_y > 0.02 * max_frame * h_)[0]
                # print("跌倒决策1", indices1)
                # ---------------------------------------------------------#
                #   跌倒决策2
                # ---------------------------------------------------------#
                head_x = points[:, 0 * 3]  # 取头部x坐标
                head_y = points[:, 0 * 3 + 1]  # 取头部y坐标
                left_ankle_x = points[:, 15 * 3]  # 取左脚部x坐标
                left_ankle_y = points[:, 15 * 3 + 1]  # 取左脚部y坐标
                right_ankle_x = points[:, 16 * 3]  # 取右脚部x坐标
                right_ankle_y = points[:, 16 * 3 + 1]  # 取右脚部y坐标
                # 计算左右脚中心x,y
                avg_ankle_x = (left_ankle_x + right_ankle_x) / 2  # 取平均脚部x坐标
                avg_ankle_y = (left_ankle_y + right_ankle_y) / 2  # 取平均脚部y坐标
                # 计算中心点与左右脚中心连线与地面的tan
                tan_angle = (avg_ankle_y - head_y) / abs(avg_ankle_x - head_x)
                # 如果tan小于等于1，则判定为跌倒
                indices2 = np.where(tan_angle <= 2)[0]
                # print("跌倒决策2:", indices2)

                # ---------------------------------------------------------#
                #   跌倒决策3
                # ---------------------------------------------------------#

                # 边框宽高比是否大于T,大于判定为跌倒
                indices3 = np.where(w_ / h_ > 0.5)[0]
                # print("跌倒决策3:", indices3)

                # ---------------------------------------------------------#
                #   跌倒决策
                # ---------------------------------------------------------#

                common1 = np.intersect1d(indices1, np.intersect1d(indices2, indices3))
                # print("跌倒决策:", common1)

                # ---------------------------------------------------------#
                #   挥拳决策
                # ---------------------------------------------------------#

                left_shoulder = points[:, 5 * 3:5 * 3 + 2]
                left_elbow = points[:, 7 * 3:7 * 3 + 2]
                left_hand = points[:, 9 * 3:9 * 3 + 2]
                right_shoulder = points[:, 6 * 3:6 * 3 + 2]
                right_elbow = points[:, 8 * 3:8 * 3 + 2]
                right_hand = points[:, 10 * 3:10 * 3 + 2]
                # 计算关节向量
                e_s = left_elbow - left_shoulder
                e_h = left_elbow - left_hand
                e_s2 = right_elbow - right_shoulder
                e_h2 = right_elbow - right_hand
                # 角度
                left_angle1 = np.arctan2(e_s[:, 0], e_s[:, 1])
                left_angle2 = np.arctan2(e_h[:, 0], e_h[:, 1])
                right_angle1 = np.arctan2(e_s2[:, 0], e_s2[:, 1])
                right_angle2 = np.arctan2(e_h2[:, 0], e_h2[:, 1])
                # 弧度转角度的转换系数
                deg_per_rad = 180 / np.pi
                # now
                left_angle = np.abs(left_angle1 - left_angle2) * deg_per_rad
                right_angle = np.abs(right_angle1 - right_angle2) * deg_per_rad

                if left_angle_pre is None and right_angle_pre is None:
                    left_angle_pre = left_angle
                    right_angle_pre = right_angle
                # now - pre
                if left_angle.shape[0] == left_angle_pre.shape[0]:
                    left_angle_vari = left_angle - left_angle_pre
                if right_angle.shape[0] == right_angle_pre.shape[0]:
                    right_angle_vari = right_angle - right_angle_pre
                # pre
                left_angle_pre = left_angle
                right_angle_pre = right_angle
                # 判断位姿
                left_pos_y = np.abs(left_hand[:, 1] - left_shoulder[:, 1])
                right_pos_y = np.abs(right_hand[:, 1] - right_shoulder[:, 1])
                if left_hand_pre is None and right_hand_pre is None:
                    left_hand_pre = left_hand
                    right_hand_pre = right_hand
                # now - pre
                # if left_hand.shape[0] == left_hand_pre.shape[0]:
                #     v_left_hand = (left_hand[0] - left_hand_pre[0]) / (e - s)
                # if right_hand.shape[0] == right_hand_pre.shape[0]:
                #     v_right_hand = right_hand[0] - right_hand_pre[0] / (e - s)

                T = w_ / h_
                if left_angle_vari.shape[0] == T.shape[0]:
                    cond1 = left_angle_vari > 15 * max_frame * T

                    # print("cond1:", cond1)
                    print("left_angle_vari",  left_angle_vari)
                    cond2 = left_pos_y < 0.2 * h_ * T
                    # print("cond2:", cond2)
                    cond3 = left_angle > 80 * T
                    # print("cond3:", cond3)
                    cond4 = right_angle_vari > 15 * max_frame * T
                    cond5 = right_pos_y < 0.2 * h_ * T
                    cond6 = right_angle > 80 * T
                    # print("left_elbow", left_elbow)
                    cond7 = left_elbow[:, 1] > head_y
                    cond8 = right_elbow[:, 1] > head_y

                # print("T", w_ / h_)

                indices4 = np.where(cond1 & cond2 & cond3 & cond7)[0]
                indices5 = np.where(cond4 & cond5 & cond6 & cond8)[0]

                union1 = np.union1d(indices4, indices5)
                if union1.shape[0] > 0:
                    for i in union1:
                        punch_num[i] += 1
                        if punch_num[i] >= 2:
                            print("!!挥拳发生!!")
                else:
                    for i in range(max_people):
                        if punch_num[i] > 0:
                            punch_num[i] -= 1

                # print("左挥拳决策", indices4)
                # print("右挥拳决策", indices5)
                print("挥拳决策", union1)

                # ---------------------------------------------------------#
                #   挥手决策
                # ---------------------------------------------------------#

                # now - pre
                if left_hand.shape[0] == left_hand_pre.shape[0]:
                    left_hand_pos_x = left_hand[:, 0] - left_hand_pre[:, 0]
                if right_hand.shape[0] == right_hand_pre.shape[0]:
                    right_hand_pos_x = right_hand[:, 0] - right_hand_pre[:, 0]
                # pre
                left_hand_pre = left_hand
                right_hand_pre = right_hand
                # 挥手判定
                if left_hand_pos_x.shape[0] == w_.shape[0]:
                    cond_left_hand = left_hand_pos_x > 0.2 * max_frame * w_
                    cond_right_hand = right_hand_pos_x > 0.2 * max_frame * w_
                left_hand_pos_y = left_hand[:, 1]
                right_hand_pos_y = right_hand[:, 1]
                if cond_right_hand.shape[0] == head_y.shape[0]:
                    cond_left_hand_y = left_hand_pos_y < head_y
                    cond_right_hand_y = right_hand_pos_y < head_y
                indices_hand = np.where((cond_left_hand & cond_left_hand_y) | (cond_right_hand & cond_right_hand_y))[0]

                # print("挥手决策", indices_hand)

                # s
                s = time.time()

                # if union1 is not None:
                #     for i in range(union1.shape[0]):
                #         index = union1[i]
                #         x = int(bboxes[index, 0])
                #         y = int(bboxes[index, 1])
                #         cv2.putText(np_img, "punch", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

            infer.draw_pose(np_img, bboxes, scores, points)

            out.write(np_img)

            np_img = cv2.resize(np_img, (640, 640))
            cv2.imshow("1", np_img)
            if cv2.waitKey(1) == ord('q'):
                break
            frame += 1
            end = time.time()
            print("fps:", 1 / (end - start))
    #         TODO:
    finally:
        if cap.isOpened():
            cap.release()
        cv2.destroyAllWindows()

    out.release()


def __stack_pic(pic, stacked_frames, is_done):
    pic = pic.reshape(1, 1, 39)
    # 如果新的开始
    if is_done:
        # Clear our stacked_frames
        stacked_frames = collections.deque([np.zeros((1, 5, 39), dtype=np.float32) for i in range(STACK_SIZE)],
                                           maxlen=STACK_SIZE)

        # Because we're in a new episode, copy the same frame 4x
        stacked_frames.append(pic)
        stacked_frames.append(pic)
        stacked_frames.append(pic)
        stacked_frames.append(pic)
        stacked_frames.append(pic)

        # Stack the frames
        stacked_pic = np.stack(stacked_frames, axis=-1)

    else:
        # Append frame to deque, automatically removes the oldest frame
        stacked_frames.append(pic)

        # Build the stacked state (first dimension specifies different frames)
        stacked_pic = np.stack(stacked_frames, axis=-1)

    return stacked_pic, stacked_frames

if __name__ == '__main__':
    # ---------------------------------------------------------#
    #   参数
    # ---------------------------------------------------------#

    capture_width = 1280
    capture_height = 720
    display_width = 1280
    display_height = 720
    framerate = 60
    flip_method = 0
    TYPE_LIST = [False, False, True, True, True, False]
    AREA_LIST = [(0, 0), (640, 640)]
    frame = 0
    is_pose = False
    is_done = True
    smoke_indices, fire_indices, indices4, indices5, indices_danger, indices_hand, union1, indices2, indices3, indices1, v_y, s, e, avg_hip_y_pre, left_angle_pre, right_angle_pre, left_hand_pre, right_hand_pre, left_angle_vari, right_angle_vari, cond1, cond2, cond3, cond7, cond4, cond5, cond6, cond8, left_hand_pos_x, right_hand_pos_x, cond_right_hand, cond_left_hand, cond_left_hand_y, cond_right_hand_y = None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None
    stacked_frames = collections.deque([np.zeros((1, 5, 39), dtype=np.float32) for i in range(STACK_SIZE)],
                                       maxlen=STACK_SIZE)

    # ---------------------------------------------------------#
    #   加载模型
    # ---------------------------------------------------------#

    infer = LoadPoseEngine('yolov8n-pose2.engine')
    infer1 = LoadEngineModel('detect7.engine')
    model = LoadLSTMEngine('models/lstm.engine')
    # model = lstm()
    # model.load('models/lstm.pth')
    print('模型加载完成！')

    # ---------------------------------------------------------#
    #   循环
    # ---------------------------------------------------------#
    # cap = cv2.VideoCapture('image/fire.mp4')
    # cap.set(cv2.CAP_PROP_FPS, 20)
    # 启用本地摄像头
    frame = 0
    cap = cv2.VideoCapture(0)
    start = time.time()
    i = 0
    while 1:
        start_i = time.time()
        frame += 1
        if frame == 1:
            frame = 0
            is_pose = True
        else:
            is_pose = False
        # 读取图片
        # np_img = cv2.imread("image/fire2.mp4")  # 替换为您的图片路径
        f, np_img = cap.read()
        np_img = cv2.resize(np_img, (640, 640))

        # 摄像头停止就关闭
        if not f:
            break

        if np_img is not None:
            i += 1
            (np_img, RES_LIST, stacked_frames, is_done) = main(infer, infer1, model, np_img, TYPE_LIST, AREA_LIST, is_pose,
                                                               stacked_frames, is_done)
            # if RES_LIST[3]:
            #     print(RES_LIST)
            # 显示图片
            bboxes, scores, points = infer(np_img)
            boxes1, scores1, idxs1 = infer1(np_img)
            scores = scores.cpu().numpy()
            bboxes = bboxes.to(torch.int32).cpu().numpy()
            points = points.cpu().numpy()  # A x 51条数据 17个点 每个点3个信息 分别是该点的x坐标，y坐标，置信度
            predlable = -1
            if points.shape[0] > 0:
                index_points = np.zeros((1, 2, 13))
                for i in range(0, 13):
                    index_points[0, 0, i] = points[0, INDEX[i] * 3] / 640
                    index_points[0, 1, i] = points[0, INDEX[i] * 3 + 1] / 640
                x_data = index_points[:, 0]
                y_data = index_points[:, 1]
                dist = np.sqrt(x_data ** 2 + y_data ** 2)
                index_points = index_points.reshape(index_points.shape[0], 26)
                index_points = np.concatenate((index_points, dist), 1)

                # TODO:
                stacked_pic, stacked_frames = __stack_pic(index_points, stacked_frames, is_done)
                input = torch.from_numpy(stacked_pic)
                input = torch.transpose(input, 1, 3)
                input = input.reshape(1, 5, 39)
                if is_done:
                    is_done = False
                model.infer(input)
                # predlable, conf = model.predict(input)
                # predlable = predlable.item()
                # pred = label2category[predlable]
                # conf = conf[0][predlable].item()
                # if conf >= 0.9:
                #     print(pred, conf)
                # else:
                #     predlable = -1
            # model.infer()
            cv2.imshow("1", np_img)
            if cv2.waitKey(1) == ord('q'):
                exit(0)
        end_i = time.time()
        # print("FPS:", 1 / (end_i - start_i))
    end = time.time()
    # 帧率
    # print("fps:", i / (end - start))
