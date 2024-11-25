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



from dataclasses import dataclass

import numpy as np

from oct_onh.loading_utils import (
    guess_ftype,
    load_dicom,
    load_image_zeiss,
)
from oct_onh.plotting import plot_enface, plot_image


@dataclass
class OCT3DVolume:

    image: np.ndarray
    res_width_mm: float = None
    res_height_mm: float = None
    res_depth_mm: float = None
    rows_y: int = None
    columns_x: int = None
    laterality: str = None
    orientation: str = None
    direction_bscan: str = None
    axial_direction: str = None
    fixation: str = None
    manufacturer: str = None
    
    def plot_enface_image(self, ax=None, flip_vertical=False):
        if flip_vertical:
            plot_enface(np.flip(self.image, axis=0), ax=ax)
        else:
            plot_enface(self.image, ax=ax)

    def plot_central_bscan(self, ax=None):
        index = len(self.image) // 2
        plot_image(self.image[index], ax=ax)
    
    @property
    def n_bscans(self) -> int:
        return len(self.image)
    
    @property
    def resolution(self) -> tuple:
        return self.res_depth_mm, self.res_height_mm, self.res_width_mm
    
    def __len__(self) -> int:
        return self.n_bscans

        
    def plot_bscan(self, bscan: int, ax=None):
        plot_image(self.image[bscan], ax=ax)

    @classmethod
    def from_file(cls, path: str):
        ftype = guess_ftype(path)
        if ftype == '.img':
            rows_y = 1024
            columns_x = 200
            image = load_image_zeiss(
                path, rows_y, columns_x)
            pixelspacing = [2 / 1024, 6/200]
            slicethickness = 6/200
            manufacturer = 'Zeiss'
        elif ftype == '.dcm':
            image, slicethickness, pixelspacing = load_dicom(
                path)
            manufacturer = 'Topcon'
        oct_im = cls(image, pixelspacing[1], pixelspacing[0],
                     slicethickness, rows_y=image.shape[1],
                     columns_x=image.shape[2], manufacturer=manufacturer)
        return oct_im
    
    
