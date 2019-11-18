# Copyright (C) tkornuta, NVIDIA AI Applications Team. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

__author__ = "Tomasz Kornuta"

import torch
import torchvision.models.detection as detection
#from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from nemo.backends.pytorch.nm import TrainableNM

from nemo.core import NeuralType, AxisType, DeviceType,\
    BatchTag, ChannelTag, HeightTag, WidthTag, ListTag, BoundingBoxTag, \
    LogProbabilityTag


class FasterRCNN(TrainableNM):
    """
        Wrapper class around the Faster R-CNN model.
    """

    @staticmethod
    def create_ports():
        input_ports = {
            # Batch of images.
            "images": NeuralType({0: AxisType(BatchTag),
                                  1: AxisType(ChannelTag, 3),
                                  2: AxisType(HeightTag),
                                  3: AxisType(WidthTag)}),
            # Batch of bounding boxes.
            "bounding_boxes": NeuralType({0: AxisType(BatchTag),
                                          1: AxisType(ListTag),
                                          2: AxisType(BoundingBoxTag)}),
            # Batch of targets.
            "targets": NeuralType({0: AxisType(BatchTag)})
        }
        output_ports = {
            "predictions": NeuralType({0: AxisType(BatchTag),
                                       1: AxisType(LogProbabilityTag)
                                       })

        }
        return input_ports, output_ports

    def __init__(self, num_classes, pretrained=False):
        """
        Creates the Faster R-CNN model.

        Args:
            num_classes: Number of output classes of the model.
            pretrained: use weights of model pretrained on COCO train2017.
        """

        super().__init__()

        # Create
        self.model = detection.fasterrcnn_resnet50_fpn(
            pretrained=True)

        # Get number of input features for the classifier.
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features

        # Replace the pre-trained head with a new one.
        self.model.roi_heads.box_predictor = \
            detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)

        self.to(self._device)

    def forward(self, images, targets=None):
        """
        Performs the forward step of the model.

        Args:
            images: Batch of images to be classified.
        """
        predictions = self.model(images, targets)
        return predictions
