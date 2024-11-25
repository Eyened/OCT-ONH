# EyeNED OCT-ONH segmentation and feature extraction
# Copyright (C) 2024  Erasmus MC

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.

# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.


import os
import torch
import numpy as np
import warnings
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor


class ONHSegmentationModel():

    def __init__(self, path=None, device='cuda'):
        torch._dynamo.config.suppress_errors = True
        # instantiate the nnUNetPredictor
        self.predictor = nnUNetPredictor(
            tile_step_size=0.5,
            use_gaussian=True,
            use_mirroring=True,
            verbose=False,
            verbose_preprocessing=False,
            device=torch.device(device),
            allow_tqdm=True)

        if path is None:
            warnings.warn("Path to the trained model is not provided. Guessing it is stored in '../model/nnUNet_plans' folder.")
            source_directory = os.path.dirname(os.path.abspath(__file__))
            path = f'{source_directory}/../model/nnUNet_plans',

        # initializes the network architecture, loads the checkpoint
        self.predictor.initialize_from_trained_model_folder(
                path,
                use_folds=(0, 1, 2, 3, 4),
                checkpoint_name='checkpoint_final.pth'
            )
        self.props = {'spacing': (999, 1, 1)}
        
    def predict(self, im_3d):
        volume_to_predict = np.expand_dims(im_3d/255, 0)
        result = self.predictor.predict_from_list_of_npy_arrays(volume_to_predict, None, self.props, None)
        return result[0]
