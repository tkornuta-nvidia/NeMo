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

import math
import torch

import nemo
import torch

from nemo.core import NeuralType, DeviceType

from nemo_cv.modules.penn_fudan_person_detection_datalayer import \
    PennFudanDataLayer, PennFudanPedestrianDataset

from nemo_cv.modules.faster_rcnn import FasterRCNN
from nemo_cv.modules.nll_loss import NLLLoss

# 0. Instantiate Neural Factory with supported backend
nf = nemo.core.NeuralModuleFactory(placement=DeviceType.CPU)

# 1. Instantiate necessary neural modules
# PennDL = PennFudanDataLayer(
#    batch_size=64,
#    shuffle=True,
#    data_folder="~/data/PennFudanPed"
# )

pfdataset = PennFudanPedestrianDataset()

# Question: how to pass 2 from DL to model?
model = FasterRCNN(2)


# 2. Describe activation's flow
# ids, imgs, boxes, targets, masks, areas, iscrowds = PennDL()
# p = model(images=imgs)


# Invoke "train" action
# nf.train([p], callbacks=[],
#         optimization_params={"num_epochs": 10, "lr": 0.001},
#         optimizer="adam")


def reduce_dict(input_dict, average=True):
    """
    Args:
        input_dict (dict): all the values will be reduced
        average (bool): whether to do average or sum
    Reduce the values in the dictionary from all processes so that all processes
    have the averaged results. Returns a dict with the same fields as
    input_dict, after reduction.
    """

    world_size = 1  # get_world_size()
    if world_size < 2:
        return input_dict
    with torch.no_grad():
        names = []
        values = []
        # sort the keys so that they are consistent across processes
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        values = torch.stack(values, dim=0)
        dist.all_reduce(values)
        if average:
            values /= world_size
        reduced_dict = {k: v for k, v in zip(names, values)}
    return reduced_dict


def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq):
    model.train()

    for image_ids, images, boxes, targets, masks, areas, iscrowds \
            in data_loader:
        images = list(image.to(device) for image in images)
        # print("images: ", type(images))
        # print("boxes: ", type(boxes))
        # print(boxes)
        # print("targets: ", type(targets))
        # print(targets)

        # targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        #boxes = [b.to(device) for b in boxes]
        # print(targets)
        #targets = [t.to(device) for t in targets]

        # We need to put this in a tuple again, as OD "framework" assumes it :]
        targets_tuple = [{"boxes": b.to(device),
                          "labels": t.to(device)} for b, t
                         in zip(boxes, targets)]
        #targets_tuple = {"boxes": boxes, "targets": targets}
        loss_dict = model.forward(images, targets_tuple)

        losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()


def collate_fn(batch):
    return tuple(zip(*batch))


# define training and validation data loaders
data_loader = torch.utils.data.DataLoader(
    pfdataset, batch_size=2, shuffle=True, num_workers=4,
    collate_fn=collate_fn)


device = torch.device(
    'cuda') if torch.cuda.is_available() else torch.device('cpu')

# move model to the right device
model.to(device)

# construct an optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005,
                            momentum=0.9, weight_decay=0.0005)


# let's train it for 10 epochs
num_epochs = 10

for epoch in range(num_epochs):
    print("Epoch: ", epoch)
    # train for one epoch, printing every 10 iterations
    train_one_epoch(model, optimizer, data_loader,
                    device, epoch, print_freq=10)

    # evaluate on the test dataset
    # evaluate(model, data_loader_test, device=device)
