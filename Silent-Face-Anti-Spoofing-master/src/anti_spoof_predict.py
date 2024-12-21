# -*- coding: utf-8 -*-
# @Time : 20-6-9 上午10:20
# @Author : zhuying
# @Company : Minivision
# @File : anti_spoof_predict.py
# @Software : PyCharm

import os
import cv2
import math
import torch
import numpy as np
import torch.nn.functional as F


from src.model_lib.MiniFASNet import MiniFASNetV1, MiniFASNetV2,MiniFASNetV1SE,MiniFASNetV2SE
from src.data_io import transform as trans
from src.utility import get_kernel, parse_model_name

MODEL_MAPPING = {
    'MiniFASNetV1': MiniFASNetV1,
    'MiniFASNetV2': MiniFASNetV2,
    'MiniFASNetV1SE':MiniFASNetV1SE,
    'MiniFASNetV2SE':MiniFASNetV2SE
}


class Detection:
    def __init__(self, deploy_path, caffemodel_path):
        self.detector = cv2.dnn.readNetFromCaffe(deploy_path, caffemodel_path)
        self.detector_confidence = 0.6

    def get_bbox(self, img):
        height, width = img.shape[0], img.shape[1]
        aspect_ratio = width / height
        if img.shape[1] * img.shape[0] >= 192 * 192:
            img = cv2.resize(img,
                           (int(192 * math.sqrt(aspect_ratio)),
                            int(192 / math.sqrt(aspect_ratio))), interpolation=cv2.INTER_LINEAR)

        blob = cv2.dnn.blobFromImage(img, 1, mean=(104, 117, 123))
        self.detector.setInput(blob, 'data')
        out = self.detector.forward('detection_out').squeeze()
        max_conf_index = np.argmax(out[:, 2])
        left, top, right, bottom = out[max_conf_index, 3]*width, out[max_conf_index, 4]*height, \
                                 out[max_conf_index, 5]*width, out[max_conf_index, 6]*height
        bbox = [int(left), int(top), int(right-left+1), int(bottom-top+1)]
        return bbox


class AntiSpoofPredict(Detection):
    def __init__(self, device_id, deploy_path, caffemodel_path):
        super().__init__(deploy_path, caffemodel_path)
        
        # Add these configuration parameters
        self.resize_size = 640 * 480  # Default resize threshold
        self.conf_threshold = 0.6  # Confidence threshold for face detection
        
        # Set up PyTorch device
        if device_id >= 0:
            self.device = torch.device("mps")
            print("Neural network using MPS")
            # self.detector.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
            # self.detector.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            
        else:
            self.device = torch.device("cpu")
            print("Neural network using CPU")
            self.detector.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
            self.detector.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    
    def _load_model(self, model_path):
        # define model
        model_name = os.path.basename(model_path)
        h_input, w_input, model_type, _ = parse_model_name(model_name)
        self.kernel_size = get_kernel(h_input, w_input,)
        self.model = MODEL_MAPPING[model_type](conv6_kernel=self.kernel_size).to(self.device)

        # load model weight
        state_dict = torch.load(model_path, map_location=self.device)
        keys = iter(state_dict)
        first_layer_name = keys.__next__()
        if first_layer_name.find('module.') >= 0:
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for key, value in state_dict.items():
                name_key = key[7:]
                new_state_dict[name_key] = value
            self.model.load_state_dict(new_state_dict)
        else:
            self.model.load_state_dict(state_dict)
        return None

    def predict(self, img, model_path):
        test_transform = trans.Compose([
            trans.ToTensor(),
        ])
        # Process on CPU, then move to GPU for neural network
        img = test_transform(img)
        img = img.unsqueeze(0).to(self.device)
        self._load_model(model_path)
        self.model.eval()
        
        with torch.no_grad():
            result = self.model.forward(img)
            result = F.softmax(result, dim=1).cpu().numpy()
        return result

    def get_bbox(self, img):
        """Optimized face detection with GPU support"""
        height, width = img.shape[:2]
        
        # Resize large images to improve performance
        max_size = 640
        if max(height, width) > max_size:
            scale = max_size / max(height, width)
            new_size = (int(width * scale), int(height * scale))
            img = cv2.resize(img, new_size)
            height, width = img.shape[:2]
        
        # Create blob and detect faces
        blob = cv2.dnn.blobFromImage(img, 1, mean=(104, 117, 123))
        self.detector.setInput(blob, 'data')
        
        # Use CUDA stream if available
        if hasattr(cv2.cuda, 'Stream'):
            try:
                stream = cv2.cuda.Stream()
                with cv2.cuda.Stream.Null():
                    out = self.detector.forward('detection_out').squeeze()
            except Exception:
                out = self.detector.forward('detection_out').squeeze()
        else:
            out = self.detector.forward('detection_out').squeeze()
        
        # Get best detection
        max_conf_index = np.argmax(out[:, 2])
        left, top, right, bottom = out[max_conf_index, 3:7] * [width, height, width, height]
        
        return [int(left), int(top), int(right-left+1), int(bottom-top+1)]

    def get_bbox_batch(self, img):
        """Get bounding boxes for all faces in the image"""
        height, width = img.shape[0:2]
        
        # Resize large images to improve performance
        if height * width > self.resize_size:
            scale = math.sqrt(height * width / self.resize_size)
            height_new = int(height / scale)
            width_new = int(width / scale)
            img_resize = cv2.resize(img, (width_new, height_new))
        else:
            img_resize = img
            scale = 1

        # Create blob and detect faces
        blob = cv2.dnn.blobFromImage(img_resize, 1, mean=(104, 117, 123))
        self.detector.setInput(blob, 'data')
        out = self.detector.forward('detection_out').squeeze()
        
        # Filter and scale boxes
        bboxes = []
        for i in range(out.shape[0]):
            confidence = out[i, 2]
            if confidence > self.conf_threshold:
                box = out[i, 3:7] * np.array([width, height, width, height])
                start_x, start_y, end_x, end_y = box.astype("int")
                # Convert to x,y,w,h format
                bboxes.append([start_x, start_y, end_x - start_x, end_y - start_y])
        
        return bboxes











