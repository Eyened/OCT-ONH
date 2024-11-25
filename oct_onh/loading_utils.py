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

import glob
import gzip
import json
import os

import numpy as np
import pydicom
from PIL import Image


def load_png_labels(path: str) -> np.ndarray:
    im = load_png(path)
    if len(im.shape) == 3:
        im = im[:, :, 0]
    return im

def load_png(path: str) -> np.ndarray:
    im = np.array(Image.open(path))
    return im

def load_binary(path: str, rows_y: int, columns_x: int) -> np.ndarray:
    with open(path, 'rb') as f:
        raw = np.frombuffer(f.read(), dtype=np.uint8)
        data = raw.reshape((-1, rows_y, columns_x), order='C')
    return data

def load_image_zeiss(path: str, rows_y: int=None, columns_x: int=None) -> np.ndarray:
    if rows_y is None or columns_x is None:
        rows_y = 1024
        columns_x = 200
    image = load_binary(path, rows_y, columns_x)
    ## Zeiss OCT images are flipped in x and y
    image = image[:, ::-1, ::-1]
    return image

def load_gzip_binary(path: str, rows_y: int, columns_x: int) -> np.ndarray:
    with gzip.open(path, 'rb') as f:
        raw = np.frombuffer(f.read(), dtype=np.uint8)
        data = raw.reshape((-1, rows_y, columns_x), order='C')
    return data

def save_gzip_binary(data: np.ndarray, path: str) -> str:
    _, ext = os.path.splitext(path)
    if ext != '.gz':
        path = path + '.gz'
    with gzip.open(path, 'wb') as f:
        f.write(data.tobytes())
    return path

def load_binary_or_gzip(path: str, rows_y: int, columns_x: int) -> np.ndarray:
    fname, ext = os.path.splitext(path)
    if ext == '.gz':
        mask = load_gzip_binary(path, rows_y, columns_x)
    else:
        mask = load_binary(path, rows_y, columns_x)
    return mask

def rotate_to_standard_orientation(array: np.ndarray, axial_direction: str, orientation: str, direction_bscan: str) -> np.ndarray:
    if orientation == 'Horizontal':
        if axial_direction == 'Up':
            array = np.flip(array, axis=0)
        if direction_bscan == 'Left':
            array = np.flip(array, axis=1)
    elif orientation == 'Vertical':
        if axial_direction == 'Left':
            array = np.rot90(array, k=3)
            if direction_bscan == 'Up':
                array = np.flip(array, axis=0)
        elif axial_direction == 'Right':
            array = np.rot90(array, k=1)
            if direction_bscan == 'Down':
                array = np.flip(array, axis=0)
    return array

def load_dicom(path):
    ds = pydicom.read_file(path)
    image = ds.pixel_array
    slicethickness = None
    pixelspacing = None
    for x in ds.iterall():
        if x.tag == (0x0028, 0x0030):
            pixelspacing = [float(i) for i in x._value]
        elif x.tag == (0x0018, 0x0050):
            slicethickness = float(x._value)
    if not slicethickness or not pixelspacing: # Zeiss OCT
        slicethickness = 6/200
        pixelspacing = [2 / 1024, 6/200]
    return image, slicethickness, pixelspacing

def load_from_directory(path, prefix=None):
    if prefix is None:
        prefix = ''
    bscan_files = glob.glob(os.path.join(path, f'{prefix}*.png'))
    if len(bscan_files) == 0:
        raise ValueError(f"No B-scans found in {path} with prefix {prefix}")
    full_prefix = '_'.join(bscan_files[0].split('_')[:-1])
    bscan_files = glob.glob(os.path.join(path, f'{full_prefix}*.png'))
    print(f"Found {len(bscan_files)} B-scans starting with {full_prefix}")
    ## Assume all bscans are there
    images = [Image.open(full_prefix + f'_{i}.png') for i in range(len(bscan_files))]
    image = np.array(images)
    return image

def guess_ftype(impath):
    filename, ext = os.path.splitext(impath)
    if ext == '.img' or ext == '.dcm':
        return ext
    elif ext == '.binary' or ext == '.gz':
        return 'binary'
    else:
        return 'png_series'
    
def load_image(impath, rows_y=None, columns_x=None):
    ftype = guess_ftype(impath)
    if ftype == '.img':
        image = load_image_zeiss(impath)
    elif ftype == '.dcm':
        image, _, _ = load_dicom(impath)
    elif ftype == 'binary':
        if rows_y is None or columns_x is None:
            raise ValueError("Rows and columns must be provided for binary images")
        image = load_binary_or_gzip(impath, rows_y, columns_x)
    elif ftype == 'png_series':
        image = load_from_directory(directory_from_identifier(impath), prefix=os.path.basename(impath).split('_')[0])
    return image
    
def load_json(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return data

def directory_from_identifier(identifier: str) -> str:
    if '[png_series_' in identifier:
        newpath = identifier.split('[')[0] + identifier.split(']')[1]
    else:
        newpath = identifier
    return os.path.dirname(newpath)
