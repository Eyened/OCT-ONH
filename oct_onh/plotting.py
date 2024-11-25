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


import matplotlib.pyplot as plt
import numpy as np


def plot_image(data: np.ndarray, ax=None, **kwargs):
    if ax is None:
        ax = plt
    ax.imshow(data, aspect='auto', cmap='gray', **kwargs)
    ax.axis('off')

def plot_thickness(data: np.ndarray, alpha=0.5, vmin=0, ax=None, **kwargs):
    if ax is None:
        ax = plt
    ax.imshow(data, aspect='auto', alpha=alpha, cmap='jet', vmin=vmin, **kwargs)
    ax.axis('off')

def plot_mask(data: np.ndarray, alpha=0.5, ax=None, **kwargs):    
    if ax is None:
        ax = plt
    ax.imshow(data, aspect='auto', alpha=alpha*(data > 0), cmap='jet', **kwargs)
    ax.axis('off')

def plot_enface(data: np.ndarray, ax=None, **kwargs):    
    enface = np.mean(data, axis=1)
    plot_image(enface, ax=ax, **kwargs)

def plot_enface_heatmap(data: np.ndarray, ax=None, **kwargs):    
    enface = np.sum(data, axis=1)
    plot_thickness(enface, ax=ax, **kwargs)