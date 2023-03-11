import os
import copy
import time
import cv2
import onnxruntime as ort
import numpy as np

# preprocess
def resize(image, target_size):
    """
    주어진 이미지를 target_size로 affine transform합니다.
    :param image: transform할 이미지
    :param target_size: 변환된 이미지의 크기 (width, height)
    :return: 변환된 이미지
    """
    # 이미지 크기와 타겟 사이즈를 확인합니다.
    h, w = image.shape[:2]
    target_w, target_h = target_size

    # 아핀 변환 매트릭스를 계산합니다.
    src_points = np.float32([[0, 0], [w, 0], [0, h]])
    dst_points = np.float32([[0, 0], [target_w, 0], [0, target_h]])
    M = cv2.getAffineTransform(src_points, dst_points)

    # 이미지를 아핀 변환합니다.
    transformed_image = cv2.warpAffine(image, M, target_size)

    return transformed_image, M

def invert_affine_transform_2box(bbox, M):
    """
    Transform a bounding box to its original scale using affine transformation matrix.
    
    Args:
        bbox (tuple): Tuple of 4 integers representing the bounding box coordinates (x1, y1, x2, y2).
        matrix (numpy.ndarray): Affine transformation matrix of shape (2, 3).
    
    Returns:
        Tuple of 4 integers representing the transformed bounding box coordinates (x1, y1, x2, y2).
    """
    matrix = cv2.invertAffineTransform(M)
    bbox = np.array(bbox, dtype=np.float32).reshape(2, 2)
    bbox = np.hstack((bbox, np.ones((2, 1))))
    transformed_bbox = np.dot(matrix, bbox.T).T
    transformed_bbox = transformed_bbox.astype(np.int32)
    x1, y1, x2, y2 = transformed_bbox.flatten()
    return x1, y1, x2, y2

def draw_box(image, x1, y1, x2, y2, color=(0, 0, 255), thickness=2):
    r = copy.deepcopy(image)
    return cv2.rectangle(r, (x1, y1), (x2, y2), color, thickness)

def draw_bbox_with_conf(img, bbox, conf, color=(0, 255, 0), thickness=2, font_scale=1.0):
    r = copy.deepcopy(img)
    x1, y1, x2, y2 = bbox
    cv2.rectangle(r, (x1, y1), (x2, y2), color=color, thickness=thickness)
    text = f"{conf:.2f}"
    text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    cv2.putText(r, text, (x1, y1 - text_size[1]), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color=color, thickness=thickness)
    return r

def distance2bbox(points, distance, max_shape=None):
    """Decode distance prediction to bounding box.

    Args:
        points (Tensor): Shape (n, 2), [x, y].
        distance (Tensor): Distance from the given point to 4
            boundaries (left, top, right, bottom).
        max_shape (tuple): Shape of the image.

    Returns:
        Tensor: Decoded bboxes.
    """
    x1 = points[:, 0] - distance[:, 0]
    y1 = points[:, 1] - distance[:, 1]
    x2 = points[:, 0] + distance[:, 2]
    y2 = points[:, 1] + distance[:, 3]
    if max_shape is not None:
        x1 = x1.clamp(min=0, max=max_shape[1])
        y1 = y1.clamp(min=0, max=max_shape[0])
        x2 = x2.clamp(min=0, max=max_shape[1])
        y2 = y2.clamp(min=0, max=max_shape[0])
    return np.stack([x1, y1, x2, y2], axis=-1)

def distance2kps(points, distance, max_shape=None):
    """Decode distance prediction to bounding box.

    Args:
        points (Tensor): Shape (n, 2), [x, y].
        distance (Tensor): Distance from the given point to 4
            boundaries (left, top, right, bottom).
        max_shape (tuple): Shape of the image.


    Returns:
        Tensor: Decoded bboxes.
    """
    preds = []
    for i in range(0, distance.shape[1], 2):
        px = points[:, i % 2] + distance[:, i]
        py = points[:, i % 2 + 1] + distance[:, i + 1]
        if max_shape is not None:
            px = px.clamp(min=0, max=max_shape[1])
            py = py.clamp(min=0, max=max_shape[0])
        preds.append(px)
        preds.append(py)
    return np.stack(preds, axis=-1)

class SCRFD:
    # backend
    # providers = ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    
    def __init__(self, model_file:str=None, nms_threshold:float=0.4):
        # Check model_file is exist
        assert os.path.exists(model_file), FileNotFoundError(f"{model_file} not found")
        self.model_file = model_file
        
        # Session
        self.session = ort.InferenceSession(self.model_file, providers=self.providers)
        self.session.set_providers(['CUDAExecutionProvider'])
        
        # NMS Constants
        assert nms_threshold > 0.0 and nms_threshold < 1.0, ValueError(f"{nms_threshold}")
        self.nms_threshold = nms_threshold
        
        # parse onnx
        self.center_cache = {}
        self.batched = False
        self._init_vars()

        
    def _init_vars(self):
        input_cfg = self.session.get_inputs()[0]
        input_shape = input_cfg.shape
        if isinstance(input_shape[2], str):
            self.input_size = None
        else:
            self.input_size = tuple(input_shape[2:4][::-1])
        input_name = input_cfg.name
        outputs = self.session.get_outputs()
        if len(outputs[0].shape) == 3:
            self.batched = True
        output_names = []
        for o in outputs:
            output_names.append(o.name)
        self.input_name = input_name
        self.output_names = output_names
        self.use_kps = False
        self._num_anchors = 1
        if len(outputs) == 6:
            self.fmc = 3
            self._feat_stride_fpn = [8, 16, 32]
            self._num_anchors = 2
        elif len(outputs) == 9:
            self.fmc = 3
            self._feat_stride_fpn = [8, 16, 32]
            self._num_anchors = 2
            self.use_kps = True
        elif len(outputs) == 10:
            self.fmc = 5
            self._feat_stride_fpn = [8, 16, 32, 64, 128]
            self._num_anchors = 1
        elif len(outputs) == 15:
            self.fmc = 5
            self._feat_stride_fpn = [8, 16, 32, 64, 128]
            self._num_anchors = 1
            self.use_kps = True
            
            
    def infer(self, p:str):
        *pred, orig_img = self.forward(p)
        det_box, conf = self.postprocess(pred)
        r = draw_bbox_with_conf(orig_img, det_box, conf )
        r = cv2.cvtColor(r, cv2.COLOR_RGB2BGR)
        cv2.imwrite('output.jpg', r)
        
        
        
########################################################

    def load_img(self, p:str):
        orig_img = cv2.imread(p, cv2.IMREAD_COLOR)
        orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB).astype(np.float32)
        
        orig_h, orig_w = orig_img.shape[:2]
        original_size = (orig_w, orig_h)
        
        resized, matrix = resize(orig_img, self.input_size)
        MEAN, STD = [127.5, 127.5, 127.5], [128.0, 128.0, 128.0]
        
        normalized_img = (resized - MEAN) / STD
        network_input = np.transpose(normalized_img, axes=[2, 0, 1])
        network_input = np.expand_dims(network_input, axis=0)
        network_input = network_input.astype(np.float32)                # double -> fp32
        
        return orig_img, matrix, network_input


    def forward(self, p:str, thresh:float=0.5):
        # empty container
        scores_list = []
        bboxes_list = []
        kpss_list = []
        
        # prepare
        orig_img, matrix, network_input = self.load_img(p)

        # forward
        net_outs = self.session.run(self.output_names, {self.input_name: network_input})

        input_height, input_width = network_input.shape[2], network_input.shape[3]
        fmc = self.fmc

        for idx, stride in enumerate(self._feat_stride_fpn):
            if self.batched:
                scores = net_outs[idx][0]
                bbox_preds = net_outs[idx+fmc][0]
                bbox_preds = bbox_preds * stride
                if self.use_kps:
                    kps_preds = net_outs[idx + fmc * 2][0] * stride
            else:
                scores = net_outs[idx]
                bbox_preds = net_outs[idx + fmc]
                bbox_preds = bbox_preds * stride
                if self.use_kps:
                    kps_preds = net_outs[idx + fmc * 2] * stride
            
            height = (-1) * (((-1) * input_height) // stride)  # Round up
            width = (-1) * (((-1) * input_width) // stride)  # Round up
            K = height * width
            
            key = (height, width, stride)
            if key in self.center_cache:
                anchor_centers = self.center_cache[key]
            else:
                anchor_centers = np.stack(np.mgrid[:height, :width][::-1],
                                          axis=-1).astype(np.float32)

                anchor_centers = (anchor_centers * stride).reshape((-1, 2))
                if self._num_anchors > 1:
                    anchor_centers = np.stack([anchor_centers] *
                                              self._num_anchors,
                                              axis=1).reshape((-1, 2))
                if len(self.center_cache) < 100:
                    self.center_cache[key] = anchor_centers

            pos_inds = np.where(scores >= thresh)[0]
            bboxes = distance2bbox(anchor_centers, bbox_preds)
            pos_scores = scores[pos_inds]
            pos_bboxes = bboxes[pos_inds]
            scores_list.append(pos_scores)
            bboxes_list.append(pos_bboxes)
            if self.use_kps:
                kpss = distance2kps(anchor_centers, kps_preds)
                kpss = kpss.reshape((kpss.shape[0], -1, 2))
                pos_kpss = kpss[pos_inds]
                kpss_list.append(pos_kpss)
        return scores_list, bboxes_list, kpss_list, matrix, orig_img
            
    def postprocess(self, network_outputs:tuple):
        scores_list, bboxes_list, kpss_list, matrix = network_outputs
        
        # stack
        scores = np.vstack(scores_list) # (4, 1)
        scores_ravel = scores.ravel()   # (1, 4)
        order = scores_ravel.argsort()[::-1]    # confidence score에 따른 오름차순 index
        bboxes = np.vstack(bboxes_list)
        
        if self.use_kps:
            kpss = np.vstack(kpss_list)
        
        pre_det = np.hstack((bboxes, scores)).astype(np.float32, copy=False)
        pre_det = pre_det[order, :]     # (M, 5), 5 = x1, y1, x2, y2, conf
        
        keep = self.nms(pre_det)
        det = pre_det[keep, :]
        
        conf = det[:, -1:].item()
        det = det[:, :4]

        if not self.batched:
            det = det.squeeze(axis=0)

        det_inv = invert_affine_transform_2box(det, matrix)
        return det_inv, conf


    def nms(self, dets):
        thresh = self.nms_threshold
        x1 = dets[:, 0]
        y1 = dets[:, 1]
        x2 = dets[:, 2]
        y2 = dets[:, 3]
        scores = dets[:, 4]

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)

            inds = np.where(ovr <= thresh)[0]
            order = order[inds + 1]

        return keep


if __name__ == '__main__':
    TEST_ONNX_WEIGHT_PATH = 'onnx_weights/scrfd_500m_240x320.onnx'
    img_path = 'sample_img_02.jpeg'
    sess = SCRFD(TEST_ONNX_WEIGHT_PATH)
    
    sess.infer(img_path)
    
    print('done')