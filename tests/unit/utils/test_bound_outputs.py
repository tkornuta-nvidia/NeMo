# ! /usr/bin/python
# -*- coding: utf-8 -*-
# =============================================================================
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

import pytest

# from nemo.core.neural_types import (
#    NeuralType,
#    NeuralTypeComparisonResult,
#    LossType,
#    AudioSignal
# )
from nemo.backends.pytorch.tutorials import MSELoss, RealFunctionDataLayer, TaylorNet
from nemo.core.neural_types import NeuralTypeComparisonResult
from nemo.utils.bound_outputs import BoundOutputs


@pytest.mark.usefixtures("neural_factory")
class TestBoundOutputs:
    @pytest.mark.unit
    def test_binding(self):
        # Create modules.
        data_source = RealFunctionDataLayer(n=100, batch_size=1)
        tn = TaylorNet(dim=4)
        loss = MSELoss()

        # Create the graph by connnecting the modules.
        x, y = data_source()
        y_pred = tn(x=x)
        lss = loss(predictions=y_pred, target=y)

        # Test default binding.
        bound_outputs = BoundOutputs()

        bound_outputs.add_defaults([x, y])
        bound_outputs.add_defaults([y_pred])
        bound_outputs.add_defaults([lss])

        assert len(bound_outputs) == 4
        defs = bound_outputs.definitions
        assert defs["x"].compare(data_source.output_ports["x"]) == NeuralTypeComparisonResult.SAME
        assert defs["y"].compare(data_source.output_ports["y"]) == NeuralTypeComparisonResult.SAME
        assert defs["y_pred"].compare(tn.output_ports["y_pred"]) == NeuralTypeComparisonResult.SAME
        assert defs["loss"].compare(loss.output_ports["loss"]) == NeuralTypeComparisonResult.SAME
        
        with pytest.raises(KeyError):
            _ = defs["lss"]

        # And now bound manually.
        bound_outputs["my_prediction"] = y_pred
        bound_outputs["my_loss"] = lss

        assert len(bound_outputs) == 2
        defs = bound_outputs.definitions
        assert defs["my_prediction"].compare(tn.output_ports["y_pred"]) == NeuralTypeComparisonResult.SAME
        assert defs["my_loss"].compare(loss.output_ports["loss"]) == NeuralTypeComparisonResult.SAME

        with pytest.raises(KeyError):
            _ = defs["x"]