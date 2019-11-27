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

from collections import OrderedDict

import torch
import torchvision.models.detection as detection
import torchvision.models.utils as utils

from torchvision.models.detection.rpn import AnchorGenerator, RPNHead, \
    RegionProposalNetwork
from torchvision.models.detection.roi_heads import RoIHeads

from torchvision.models.detection.transform import \
    GeneralizedRCNNTransform

from torchvision.ops import MultiScaleRoIAlign

from torchvision.models.detection.faster_rcnn import TwoMLPHead, \
    FastRCNNPredictor

#from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from nemo.backends.pytorch.nm import TrainableNM

from nemo.core import NeuralType, AxisType, DeviceType,\
    BatchTag, ChannelTag, HeightTag, WidthTag, ListTag, BoundingBoxTag, \
    LogProbabilityTag

model_urls = {
    'fasterrcnn_resnet50_fpn_coco':
    'https://download.pytorch.org/models/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth',
}


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
            "targets": NeuralType({0: AxisType(BatchTag)}),
            # "Artificial" variable - tensor storing numbers of objects.
            "num_objects": NeuralType({0: AxisType(BatchTag)})
        }
        output_ports = {
            "predictions": NeuralType({0: AxisType(BatchTag),
                                       1: AxisType(LogProbabilityTag)
                                       })

        }
        return input_ports, output_ports

    def __init__(self, num_classes=91, progress=True, pretrained=False,
                 pretrained_backbone=True,
                 # transform parameters
                 min_size=800, max_size=1333,
                 # RPN parameters
                 rpn_pre_nms_top_n_train=2000, rpn_pre_nms_top_n_test=1000,
                 rpn_post_nms_top_n_train=2000, rpn_post_nms_top_n_test=1000,
                 rpn_nms_thresh=0.7,
                 rpn_fg_iou_thresh=0.7, rpn_bg_iou_thresh=0.3,
                 rpn_batch_size_per_image=256, rpn_positive_fraction=0.5,
                 # Box parameters
                 box_score_thresh=0.05, box_nms_thresh=0.5,
                 box_detections_per_img=100,
                 box_fg_iou_thresh=0.5, box_bg_iou_thresh=0.5,
                 box_batch_size_per_image=512, box_positive_fraction=0.25
                 ):
        """
        Creates the Faster R-CNN model.

        Args:
            num_classes: Number of output classes of the model.
            pretrained: use weights of model pretrained on COCO train2017.
        """

        super().__init__()

        # Create the model.
        if pretrained:
            # no need to download the backbone if pretrained is set
            pretrained_backbone = False

        # Create the ResNet+FPN "backbone".
        backbone = detection.backbone_utils.resnet_fpn_backbone(
            'resnet50', pretrained_backbone)

        # Create the other pieces of the model.
        out_channels = backbone.out_channels

        # if rpn_anchor_generator is None:
        anchor_sizes = ((32,), (64,), (128,), (256,), (512,))
        aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
        rpn_anchor_generator = AnchorGenerator(
            anchor_sizes, aspect_ratios
        )

        # if rpn_head is None:
        rpn_head = RPNHead(
            out_channels, rpn_anchor_generator.num_anchors_per_location()[
                0]
        )

        rpn_pre_nms_top_n = dict(
            training=rpn_pre_nms_top_n_train, testing=rpn_pre_nms_top_n_test)
        rpn_post_nms_top_n = dict(
            training=rpn_post_nms_top_n_train, testing=rpn_post_nms_top_n_test)

        rpn = RegionProposalNetwork(
            rpn_anchor_generator, rpn_head,
            rpn_fg_iou_thresh, rpn_bg_iou_thresh,
            rpn_batch_size_per_image, rpn_positive_fraction,
            rpn_pre_nms_top_n, rpn_post_nms_top_n, rpn_nms_thresh)

        # if box_roi_pool is None:
        box_roi_pool = MultiScaleRoIAlign(
            featmap_names=[0, 1, 2, 3],
            output_size=7,
            sampling_ratio=2)

        # if box_head is None:
        resolution = box_roi_pool.output_size[0]
        representation_size = 1024
        box_head = TwoMLPHead(
            out_channels * resolution ** 2,
            representation_size)

        # if box_predictor is None:
        representation_size = 1024
        box_predictor = FastRCNNPredictor(
            representation_size,
            num_classes)

        roi_heads = RoIHeads(
            # Box
            box_roi_pool, box_head, box_predictor,
            box_fg_iou_thresh, box_bg_iou_thresh,
            box_batch_size_per_image, box_positive_fraction,
            None,
            box_score_thresh, box_nms_thresh, box_detections_per_img)

        # if image_mean is None:
        image_mean = [0.485, 0.456, 0.406]
        # if image_std is None:
        image_std = [0.229, 0.224, 0.225]
        transform = GeneralizedRCNNTransform(
            min_size, max_size, image_mean, image_std)

        #self.model = detection.FasterRCNN(backbone, num_classes)

        #super(FasterRCNN, self).__init__(backbone, rpn, roi_heads, transform)

        self.transform = transform
        self.backbone = backbone
        self.rpn = rpn
        self.roi_heads = roi_heads

        # if pretrained:
        #    state_dict = utils.load_state_dict_from_url(
        #        model_urls['fasterrcnn_resnet50_fpn_coco'],
        #        progress=progress)
        #    self.model.load_state_dict(state_dict)

        # Get number of input features for the classifier.
        in_features = self.roi_heads.box_predictor.cls_score.in_features

        # Replace the pre-trained head with a new one.
        self.roi_heads.box_predictor = \
            detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)

        self.to(self._device)

    def forward(self, images, bounding_boxes, targets, num_objects):
        """
        Performs the forward step of the model.

        Args:
            images: Batch of images to be classified.
        """
        #print("Faster R-CNN forward:")
        # We need to put this in a tuple again, as OD "framework" assumes it :]

        # Unstack tensors with boxes and target, removing the "padded objects".
        bboxes_padded = torch.unbind(bounding_boxes, dim=0)
        targets_padded = torch.unbind(targets, dim=0)

        # Unpad bounding boxes.
        bboxes_unpadded = []
        for i in range(len(bounding_boxes)):
            bboxes_unpadded.append(bboxes_padded[i][0:num_objects[i], :])

        # Unpad targets.
        targets_unpadded = []
        for i in range(len(targets_padded)):
            targets_unpadded.append(targets_padded[i][0:num_objects[i]])

        targets_tuple = [{"boxes": b, "labels": t} for b, t
                         in zip(bboxes_unpadded, targets_unpadded)]

        # THE PROPPER forward pass.
        #loss_dict = self.model(images, targets_tuple)

        if self.training and targets_tuple is None:
            raise ValueError("In training mode, targets should be passed")
        original_image_sizes = [img.shape[-2:] for img in images]

        # Preprocess the images.
        images, targets_tuple = self.transform(images, targets_tuple)

        # Extract the features.
        features = self.backbone(images.tensors)

        if isinstance(features, torch.Tensor):
            features = OrderedDict([(0, features)])

        # Calculate the region proposals.
        proposals, proposal_losses = self.rpn(images, features, targets_tuple)

        # Calculate the regions.
        detections, detector_losses = self.roi_heads(
            features, proposals, images.image_sizes, targets_tuple)

        # Empty!!! No detections in "training" mode.
        # print(detections)

        # Postprocess the images.
        detections = self.transform.postprocess(
            detections, images.image_sizes, original_image_sizes)

        loss_dict = {}
        loss_dict.update(detector_losses)
        loss_dict.update(proposal_losses)

        # if self.training:
        #    return losses

        # Sum losses.
        losses = sum(loss for loss in loss_dict.values())

        print("Loss = ", losses.item())

        return losses
