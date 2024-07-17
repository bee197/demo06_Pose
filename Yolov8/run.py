import collections
import math
from collections import namedtuple, OrderedDict
import time
import cv2
import numpy as np
import tensorrt as trt
import torch
from torchvision import ops
from utils import format_img, gstreamer_pipeline
from utils1 import format_img, draw_on_src, gstreamer_pipeline
from main import LoadEngineModel
from model import lstm

KPS_COLORS = [[0, 255, 0], [0, 255, 0], [0, 255, 0], [0, 255, 0], [0, 255, 0],
              [255, 128, 0], [255, 128, 0], [255, 128, 0], [255, 128, 0],
              [255, 128, 0], [255, 128, 0], [51, 153, 255], [51, 153, 255],
              [51, 153, 255], [51, 153, 255], [51, 153, 255], [51, 153, 255]]

SKELETON = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13],
            [6, 7], [6, 8], [7, 9], [8, 10], [9, 11], [2, 3], [1, 2], [1, 3],
            [2, 4], [3, 5], [4, 6], [5, 7]]

INDEX = [0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]

LIMB_COLORS = [[51, 153, 255], [51, 153, 255], [51, 153, 255], [51, 153, 255],
               [255, 51, 255], [255, 51, 255], [255, 51, 255], [255, 128, 0],
               [255, 128, 0], [255, 128, 0], [255, 128, 0], [255, 128, 0],
               [0, 255, 0], [0, 255, 0], [0, 255, 0], [0, 255, 0], [0, 255, 0],
               [0, 255, 0], [0, 255, 0]]
label2category = {
    0: '挥手',  # 挥手
    1: '跌倒',  # 跌倒
    2: '打/拍打',  # 打/拍打
}
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
        print("out_prob", out_prob.shape)
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


def run(infer, infer1, model, np_img, TYPE_LIST, AREA_LIST, is_pose, stacked_frames, is_done):
    bboxes, scores, points = infer(np_img)
    scores = scores.cpu().numpy()
    bboxes = bboxes.to(torch.int32).cpu().numpy()
    points = points.cpu().numpy()  # A x 51条数据 17个点 每个点3个信息 分别是该点的x坐标，y坐标，置信度
    infer.draw_pose(np_img, bboxes, scores, points)

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
        predlable, conf = model.predict(input)
        predlable = predlable.item()
        pred = label2category[predlable]
        conf = conf[0][predlable].item()
        if conf >= 0.9:
            print(pred, conf)
        else:
            predlable = -1

    boxes1, scores1, idxs1 = infer1(np_img)
    boxes1, scores1, idxs1 = boxes1.cpu().numpy(), scores1.cpu().numpy(), idxs1.cpu().numpy()


if __name__ == '__main__':

    # ---------------------------------------------------------#
    #   参数
    # ---------------------------------------------------------#

    TYPE_LIST = [False, False, True, True, True, False]
    AREA_LIST = [(0, 0), (640, 640)]
    frame = 0
    is_pose = False
    is_done = True
    stacked_frames = collections.deque([np.zeros((1, 5, 39), dtype=np.float32) for i in range(STACK_SIZE)],
                                       maxlen=STACK_SIZE)

    # ---------------------------------------------------------#
    #   加载模型
    # ---------------------------------------------------------#

    infer = LoadPoseEngine('yolov8n-pose2.engine')
    infer1 = LoadEngineModel('detect7.engine')
    model = lstm()
    model.load('models/lstm.pth')
    print('模型加载完成！')

    # ---------------------------------------------------------#
    #   循环
    # ---------------------------------------------------------#
    cap = cv2.VideoCapture('image/挥拳5.mp4')
    # cap.set(cv2.CAP_PROP_FPS, 20)
    # 启用本地摄像头
    # cap = cv2.VideoCapture(0)
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
            # (np_img, RES_LIST, stacked_frames, is_done) = run(infer, infer1, model, np_img, TYPE_LIST, AREA_LIST, is_pose,
            #                                                    stacked_frames, is_done)
            # if RES_LIST[3]:
            #     print(RES_LIST)
            boxes, scores, idxs = infer(np_img)
            boxes1, scores1, idxs1 = infer1(np_img)
        end_i = time.time()
        # print("FPS:", 1 / (end_i - start_i))
    end = time.time()
    # 帧率
    # print("fps:", i / (end - start))
