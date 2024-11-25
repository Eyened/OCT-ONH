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
from oct_onh.plotting import plot_image


@dataclass

class _OCTBScan():

    imagedata: np.ndarray = None
    res_width_mm: float = None
    res_height_mm: float = None
    rows_y: int = None
    columns_x: int = None
    laterality: str = None
    
    @property
    def resolution(self) -> tuple:
        return (self.res_height_mm, self.res_width_mm)
    
    def __len__(self) -> int:
        return len(self.image)
        
    def plot(self):
        plot_image(self.image)

    @property
    def image(self):
        return self.imagedata


@dataclass
class OCTBScan(_OCTBScan):
    
    @property
    def resolution(self) -> tuple:
        return (self.res_height_mm, self.res_width_mm)
    
    def __len__(self) -> int:
        return len(self.image)
        
    def plot(self):
        plot_image(self.image)

