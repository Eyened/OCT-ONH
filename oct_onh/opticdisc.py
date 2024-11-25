
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
import warnings
from dataclasses import dataclass
from functools import cached_property
from typing import List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import lstsq
from scipy.ndimage import binary_fill_holes, distance_transform_edt
from skimage import measure
from skimage.morphology import binary_closing, skeletonize

from oct_onh.loading_utils import load_binary_or_gzip
from oct_onh.octbscan import OCTBScan
from oct_onh.octvolume import OCT3DVolume
from oct_onh.plotting import plot_enface, plot_enface_heatmap, plot_image


class OpticDiscLabels():
    BM_LABELS = [2, 3, 6, 7, 8]
    VESSELS = [5]
    RNFL = [1]
    RNFL_AND_VESSELS = [1, 5]
    LC = [4]
    PPA_A = [6, 7]
    PPA_B = [8]
    PPA_G = [9]

def read_optic_disc_image(imagepath, maskpath):
    _ , ext = os.path.splitext(imagepath)
    if ext == '.dcm' or ext == '.img':
        return OpticDiscVolume.from_file(imagepath, maskpath)
    elif os.path.isdir(imagepath):
        return OpticDiscOCTEnfaceGroup.from_file(imagepath, maskpath)
    raise ValueError(f'Unknown file format {ext}')

@dataclass
class OpticDiscBscan():

    bscan: OCTBScan
    mask: np.ndarray = None
    labels = OpticDiscLabels

    @property
    def image(self) -> np.ndarray:
        return self.bscan.image

    @property
    def resolution(self) -> np.ndarray:
        return self.bscan.resolution
    
    @property
    def res_height_mm(self) -> float:
        return self.bscan.res_height_mm
    
    @property
    def res_width_mm(self) -> float:
        return self.bscan.res_width_mm
    
    @property
    def laterality(self) -> float:
        return self.bscan.laterality
    
    def get_all_features(self):
        raise NotImplementedError('Feature extraction not implemented')

@dataclass
class OpticDiscRadialBScan(OpticDiscBscan):
    
    startpoint: tuple = None
    endpoint: tuple = None
    rnfl_mask: np.ndarray = None
    bmo_center: tuple = None
    res_y_enface: float = None
    res_x_enface: float = None
    bmo_startpoint: tuple = None
    bmo_endpoint: tuple = None

    @property
    def angle(self) -> float:
        return np.arctan2((self.endpoint[1] - self.startpoint[1])*self.res_y_enface,
                          (self.endpoint[0] - self.startpoint[0])*self.res_x_enface)
    @property
    def angle_left(self) -> float:
        return (self.angle + np.pi) % (np.pi*2)
    
    @property
    def angle_right(self) -> float:
        return (self.angle + np.pi*2) % (np.pi*2)
    
    def get_all_features(self):
        results = {}
        results = results | self.detect_bm_opening()
        results = results | self.get_minimal_rim_width()
        return results

    def get_rnfl_mask(self):
        if self.rnfl_mask is None:
            rnfl_map = np.isin(self.mask, self.labels.RNFL).astype('uint8')
            vessels_map = np.isin(self.mask, self.labels.VESSELS).astype('uint8')
            background = np.isin(self.mask, self.labels.RNFL_AND_VESSELS).astype('uint8')

            background_filled = binary_fill_holes(background)
            dist_rnfl = distance_transform_edt(1-rnfl_map)
            dist_background = distance_transform_edt(background_filled)
            rnfl_map = (binary_fill_holes(rnfl_map) + (vessels_map & (dist_rnfl < dist_background))) > 0
            self.rnfl_mask = rnfl_map
        return self.rnfl_mask

    def detect_bm_opening(self):
        mask = self.mask
        all_bm = np.isin(mask, self.labels.BM_LABELS).astype('uint8')
        self.all_bm = all_bm

        enface_bmo_presence = np.where(
            np.sum(all_bm, axis=0) > 0, 0, 1).astype('uint8')
        enface_bmo_presence = binary_closing(enface_bmo_presence, np.ones(5))

        bmo_connected_components, n = measure.label(
            enface_bmo_presence, background=0, return_num=True)
        if n == 0 or np.sum(enface_bmo_presence) < 2:
            bmo_width = 0
            bmo_width_surface = 0
            self.bmo_center = [0, 0]
        else:
            sizes = [np.sum(np.where(bmo_connected_components == i, 1, 0))
                     for i in range(np.max(bmo_connected_components)+1)]
            sizes[0] = len(enface_bmo_presence) # never select background
            enface_bmo_presence_corrected = np.where(
                bmo_connected_components == np.argsort(sizes)[-2], 1, 0)
            self.bmo_mask = enface_bmo_presence_corrected

            bmo_start_x = np.argmax(enface_bmo_presence_corrected) - 1
            bmo_end_x = len(enface_bmo_presence_corrected)\
                - np.argmax(enface_bmo_presence_corrected[::-1]) + 1
            if bmo_end_x > len(enface_bmo_presence_corrected):
                bmo_width = 0
                bmo_width_surface = 0
                self.bmo_center = [0, 0]
                return {}
            bmo_width = (bmo_end_x - bmo_start_x) * self.res_height_mm

            bmo_start_y = np.argmax(all_bm[:, bmo_start_x])
            bmo_end_y = np.argmax(all_bm[:, bmo_end_x])

            startpoint = [bmo_start_y, bmo_start_x]\
                * np.array(self.resolution)
            endpoint = [bmo_end_y, bmo_end_x] * np.array(self.resolution)

            self.bmo_center = [(bmo_end_y + bmo_start_y) / 2,
                               (bmo_end_x + bmo_start_x) / 2]

            bmo_width_surface = np.linalg.norm(startpoint - endpoint)
            self.bmo_startpoint = (bmo_start_x, bmo_start_y)
            self.bmo_endpoint = (bmo_end_x, bmo_end_y)
            self.bmo_start_x = bmo_start_x
            self.bmo_end_x = bmo_end_x
            self.bmo_start_y = bmo_start_y
            self.bmo_end_y = bmo_end_y
        return {
            'BMO width 2D flattened (mm)': bmo_width,
            'BMO width 2D (mm)': bmo_width_surface
        }

    def get_minimal_rim_width(self):
        left_top_point, right_top_point, dist_left, dist_right = self.get_mrw_points()
        return {
            'Minimal rim width left (mm)': dist_left,
            'Minimal rim width right (mm)': dist_right
        }
    
    def get_mrw_points(self):
        if not self.bmo_startpoint or not self.bmo_endpoint:
            self.detect_bm_opening()
        if not self.bmo_startpoint or not self.bmo_endpoint:
            return {}
        rnfl_map = self.get_rnfl_mask()
        rnfl_top = np.argmax(rnfl_map, axis=0)
        grid = np.indices(self.mask.shape)
        dist_y = np.abs(grid[0] - self.bmo_startpoint[1]) * self.res_height_mm
        dist_x = np.abs(grid[1] - self.bmo_startpoint[0]) * self.res_width_mm
        dist = np.sqrt(dist_y**2 + dist_x**2)

        dist_values = dist[rnfl_top, range(rnfl_top.shape[0])]

        dist_startpoint_x = np.argmin(dist_values)
        dist_startpoint = np.min(dist_values)
        dist_startpoint_y = rnfl_top[dist_startpoint_x]

        dist_y = np.abs(grid[0] - self.bmo_endpoint[1]) * self.res_height_mm
        dist_x = np.abs(grid[1] - self.bmo_endpoint[0]) * self.res_width_mm
        dist = np.sqrt(dist_y**2 + dist_x**2)

        dist_values = dist[rnfl_top, range(rnfl_top.shape[0])]
        dist_endpoint_x = np.argmin(dist_values)
        dist_endpoint = np.min(dist_values)
        dist_endpoint_y = rnfl_top[dist_endpoint_x]
        return (dist_startpoint_x, dist_startpoint_y), (dist_endpoint_x, dist_endpoint_y), dist_startpoint, dist_endpoint
    
    def plot_mrw(self, ax:plt.Axes=None):
        if ax is None:
            _, ax = plt.subplots()

        left_top_point, right_top_point, dist_left, dist_right = self.get_mrw_points()
        
        aspect = self.image.shape[0]*self.resolution[0] / (self.image.shape[1] * self.resolution[1])

        ax.imshow(self.image, aspect=aspect, cmap='gray')
        ax.imshow(self.mask, alpha=0.5*(self.mask > 0), cmap='viridis', aspect=aspect, vmax=6)
        ax.plot(
                [self.bmo_startpoint[0], left_top_point[0]],
                [self.bmo_startpoint[1], left_top_point[1]], 'w-')
        ax.annotate(
            f'{dist_left:.3f} mm',
            (left_top_point[0] + 2, left_top_point[0] + 2), color='white')
        ax.plot(
            [self.bmo_endpoint[0], right_top_point[0]],
            [self.bmo_endpoint[1], right_top_point[1]], 'w-')
        ax.annotate(
            f'{dist_right:.3f} mm',
            (right_top_point[0] + 2, right_top_point[1] + 2), color='white')
        ax.axis('off')

@dataclass
class OpticDiscCircularBScan(OpticDiscBscan):
    
    center: tuple = None
    radius: float = None
    start_angle: float = None
    real_diameter: float = None

    @cached_property
    def angles_from_fovea(self):
        return np.linspace(0, 2*np.pi, self.image.shape[1])
    
    @property
    def angles(self):
        if self.laterality == 'L':
            return ((self.start_angle + self.angles_from_fovea)) % (2*np.pi)
        else:
            return ((self.start_angle - self.angles_from_fovea)) % (2*np.pi)
    
    @property
    def angles_from_top(self):
        return (2.5 * np.pi - self.angles) % (2*np.pi)

    @cached_property
    def rnfl_mask(self):

        rnfl_map = np.isin(self.mask, self.labels.RNFL).astype('uint8')
        vessels_map = np.isin(self.mask, self.labels.VESSELS).astype('uint8')
        background = np.isin(self.mask, self.labels.RNFL_AND_VESSELS).astype('uint8')
        background_filled = binary_fill_holes(background)
        dist_rnfl = distance_transform_edt(1-rnfl_map)
        dist_background = distance_transform_edt(background_filled)
        rnfl_map = (binary_fill_holes(rnfl_map) + (vessels_map & (dist_rnfl < dist_background))) > 0
        return rnfl_map
    
    def get_all_features(self):
        results = {}
        results = results | self.get_rnfl_thickness_circular()
        return results
    
    def get_clock_hours(self, values):
        results = {}
        for hour in range(12):
            angle = hour * (np.pi / 6)
            start = (angle - (np.pi / 12) + 2*np.pi) % (2*np.pi)
            end = angle + (np.pi / 12)
            if start > end:
                indices = np.logical_or(
                    self.angles_from_top > start, self.angles_from_top <= end)
            else:
                indices = np.logical_and(
                    self.angles_from_top > start, self.angles_from_top <= end)
            measurements = values[indices]
            results[hour] =measurements

        return results
    
    def get_quadrants(self, values):
        results = {}
        for quadrant in range(4):
            angle = quadrant * (np.pi / 2)
            start = (angle - (np.pi / 4) + 2*np.pi) % (2*np.pi)
            end = angle + (np.pi / 4)

            if start > end:
                indices = np.logical_or(
                    self.angles_from_top > start, self.angles_from_top <= end)
            else:
                indices = np.logical_and(
                    self.angles_from_top > start, self.angles_from_top <= end)
            measurements = values[indices]
            results[quadrant] = measurements
        return results

    def get_tsnit_segments(self, values):
        results = {}
        angles = self.angles_from_fovea * 360 / (2*np.pi)
        for area, start, end in zip(
                    ['T', 'TS', 'NS', 'N', 'NI', 'TI'],
                    [310, 40, 80, 120, 230, 270],
                    [40, 80, 120, 230, 270, 310]
                ):
            # In the Heidelberg circular scan, the left eye runs
            # from 0 (right) anti-clockwise. The right eye goes from
            # 180 degrees (left) clockwise. The correction is defined
            # as the angle at which the fovea lies, so the angles are shifted
            # towards the fovea
            angles_from_fovea = (angles) %\
                360

            if start > end:
                indices = np.logical_or(
                    angles_from_fovea > start, angles_from_fovea <= end)
            else:
                indices = np.logical_and(
                    angles_from_fovea > start, angles_from_fovea <= end)
            measurements = values[indices]
            results[area] = measurements
        return results

    @cached_property
    def rnfl_thickness(self):
        return np.sum(self.rnfl_mask, axis=0)*self.res_height_mm

    def get_rnfl_thickness_circular(self):
        results = {}

        measurmements = self.get_tsnit_segments(self.rnfl_thickness)
        for key, value in measurmements.items():
            results[f'RNFL thickness d={self.real_diameter:.1f} mm segment {key} (mm) mean'] = np.mean(value)
            results[f'RNFL thickness d={self.real_diameter:.1f} mm segment {key} (mm) n points'] = len(value)


        measurmements = self.get_clock_hours(self.rnfl_thickness)
        for key, value in measurmements.items():
            results[f'RNFL thickness d={self.real_diameter:.1f} mm hour {key} (mm) mean'] = np.mean(value)
            results[f'RNFL thickness d={self.real_diameter:.1f} mm hour {key} (mm) n points'] = len(value)

        measurmements = self.get_quadrants(self.rnfl_thickness)
        for key, value in measurmements.items():
            results[f'RNFL thickness d={self.real_diameter:.1f} mm quadrant {key} (mm) mean'] = np.mean(value)
            results[f'RNFL thickness d={self.real_diameter:.1f} mm quadrant {key} (mm) n points'] = len(value)
        
        return results
    
    @property
    def x(self):
        return self.center[0] + self.radius * np.cos(self.angles)

    @property
    def y(self):
        return self.center[1] - self.radius * np.sin(self.angles)

    def get_tsnit_locations(self):
        xvals = self.get_tsnit_segments(self.x)
        yvals = self.get_tsnit_segments(self.y)
        return xvals, yvals
    
    def get_clock_hour_locations(self):
        xvals = self.get_clock_hours(self.x)
        yvals = self.get_clock_hours(self.y)
        return xvals, yvals
    
    def get_quadrant_locations(self):
        xvals = self.get_quadrants(self.x)
        yvals = self.get_quadrants(self.y)
        return xvals, yvals
    
    def get_circular_positioning(self, regiontype='tsnit'):
        if regiontype=='tsnit':
            return self.get_tsnit_locations()
        elif regiontype=='clockhours':
            return self.get_clock_hour_locations()
        elif regiontype=='quadrants':
            return self.get_quadrant_locations()
    
    def get_bmo_ellipse(self, contour_points):
        ellipse = cv2.fitEllipse(contour_points)
        return ellipse

class AbstractOpticDiscOCT():

    @property
    def enface_image(self):
        raise NotImplementedError('Enface image not implemented')

    def get_all_features(self):
        raise NotImplementedError('Feature extraction not implemented')
    
    def get_positioning(self, angles, values, regiontype='tsnit') -> dict:
        if regiontype=='tsnit':
            return self.get_tsnit_segments(angles, values)
        elif regiontype=='clockhours':
            return self.get_clock_hours(angles, values)
        elif regiontype=='quadrants':
            return self.get_quadrants(angles, values)

    def get_tsnit_segments(self, angles, values) -> dict:
        results = {}
        for area, start, end in zip(
                    ['T', 'TS', 'NS', 'N', 'NI', 'TI'],
                    [310, 40, 80, 120, 230, 270],
                    [40, 80, 120, 230, 270, 310]
                ):
            if self.laterality == 'L':
                angles_degrees = 360 - (angles / (2*np.pi) * 360)
            else:
                angles_degrees = 180 + (angles / (2*np.pi) * 360)
            angles_degrees = angles_degrees % 360
            if start > end:
                indices = np.logical_or(
                    angles_degrees > start, angles_degrees <= end)
            else:
                indices = np.logical_and(
                    angles_degrees > start, angles_degrees <= end)
            measurements = values[indices]
            results[area] = measurements
        return results
    
    def get_quadrants(self, angles, values) -> dict:
        results = {}
        for quadrant in range(4):
            angle = quadrant * (np.pi / 2) - (np.pi/2)
            # Prevent negative numbers
            start = (angle + 2*np.pi - (np.pi / 4)) % (2*np.pi)
            end = (angle + (np.pi / 4)) % (2*np.pi)
            ## We add a small increment to the end to prevent rounding errors
            ## when the angle is exactly at the border of two intervals
            if start > end:
                indices = np.logical_or(angles > start, angles <= (end+0.001))
            else:
                indices = np.logical_and(angles > start, angles <= (end+0.001))
            measurements = values[indices]
            results[quadrant] = measurements
        return results
    
    def get_clock_hours(self, angles, values) -> dict:
        results = {}
        all_indices = np.zeros(angles.shape, dtype=bool)
        for hour in range(12):
            angle = (hour * (np.pi / 6) - (np.pi/2)) % (2*np.pi)
            # Prevent negative numbers
            start = (angle + 2*np.pi - (np.pi / 12)) % (2*np.pi)
            end = (angle + 2*np.pi + (np.pi / 12)) % (2*np.pi)

            ## We add a small increment to the end to prevent rounding errors
            ## when the angle is exactly at the border of two intervals
            if start > end:
                indices = np.logical_or(angles > start, angles <= (end+0.001))
            else:
                indices = np.logical_and(angles > start, angles <= (end+0.001))
            measurements = values[indices]
            all_indices[indices] = True
            results[hour] = measurements
        return results
    
    def get_mrw_raw(self) -> Tuple[np.ndarray, np.ndarray]:
        extracted_angles = []
        rim_widths = []
        bscans = self.get_radial_bscans()
        for bscan in bscans:
            this_result = bscan.get_minimal_rim_width()
            if len(this_result) > 0:
                rim_widths.append(this_result['Minimal rim width left (mm)'])
                extracted_angles.append(bscan.angle_left)
                rim_widths.append(this_result['Minimal rim width right (mm)'])
                extracted_angles.append(bscan.angle_right)
            else:
                warnings.warn(f'Minimal rim width failed for angle {bscan.angle}')
        extracted_angles = np.array(extracted_angles)
        rim_widths = np.array(rim_widths)
        return extracted_angles, rim_widths
       
    def get_minimal_rim_width(self):
        angles, measurements = self.get_mrw_raw()
        results = {}
        clock_hours = self.get_clock_hours(angles, measurements)
        for key, values in clock_hours.items():
            results[f'Minimal rim width hour {key} mean (mm)'] = np.mean(values)
            results[f'Minimal rim width hour {key} n measurements'] = len(values)
        quadrants = self.get_quadrants(angles, measurements)
        for key, values in quadrants.items():
            results[f'Minimal rim width quadrant {key} mean (mm)'] = np.mean(values)
            results[f'Minimal rim width quadrant {key} n measurements'] = len(values)
        segments = self.get_tsnit_segments(angles, measurements)
        for key, values in segments.items():
            results[f'Minimal rim width region {key} mean (mm)'] = np.mean(values)
            results[f'Minimal rim width region {key} n measurements'] = len(values)
        results[f'Minimal rim width global mean (mm)'] =\
            np.mean(measurements)
        return results
    
    def get_radial_bscans() -> List[OpticDiscRadialBScan]:
        raise NotImplementedError('Radial bscans not implemented')
    
    def plot_radial_positioning(self, regiontype='tsnit', ax=None):
        if ax is None:
            _, ax = plt.subplots()
        points_x = []
        points_y = []
        angles = []
        bscans = self.get_radial_bscans()
        for b in bscans:
            angles.append(b.angle_left)
            points_x.append(b.startpoint[0])
            points_y.append(b.startpoint[1])
            angles.append(b.angle_right)
            points_x.append(b.endpoint[0])
            points_y.append(b.endpoint[1])
        angles = np.array(angles)
        points_x = np.array(points_x)
        points_y = np.array(points_y)
        x_dict = self.get_positioning(angles, points_x, regiontype)
        y_dict = self.get_positioning(angles, points_y, regiontype)
        
        plot_image(self.enface_image, ax=ax)
        for k in x_dict.keys():
            xvals = x_dict[k]
            yvals = y_dict[k]
            ax.scatter(xvals, yvals, label=k, s=5)
            ax.legend()

    def get_bmo_ellipse(self, contour_points):
        ellipse = cv2.fitEllipse(contour_points)
        return ellipse

    def plot_mrw_bscans(self, savepath=None):
        bscans = self.get_radial_bscans()
        fig, axes = plt.subplots(6, 4, figsize=(15, 20))
        for i, (bscan, ax) in enumerate(zip(bscans, axes.flatten())):
            bscan.plot_mrw(ax=ax)
        plt.tight_layout()
        if savepath:
            plt.savefig(savepath)
        plt.show()
        plt.close()

    def plot_mrw(self, ax=None):
        if ax is None:
            _, ax = plt.subplots()
        angles, measurements = self.get_mrw_raw()
        ax.scatter(angles, measurements)

@dataclass
class OpticDiscVolume(OCT3DVolume, AbstractOpticDiscOCT):

    mask: np.ndarray = None
    labels = OpticDiscLabels
    rnfl_mask = None
    bmo_center = None

    @classmethod
    def from_file_with_mask(cls, path: str, maskpath: str):
        octim = super().from_file(path)

        mask = load_binary_or_gzip(maskpath, octim.rows_y, octim.columns_x)
        if octim.manufacturer == 'Zeiss':
            mask = mask[:, :, ::-1]
        octim.mask = mask
        return octim
    
    @classmethod
    def from_file_with_model(cls, path: str, model=None):
        octim = super().from_file(path)

        if model is None:
            from oct_onh.model import ONHSegmentationModel
            model = ONHSegmentationModel()
        mask = model.predict(octim.image)
        octim.mask = mask
        return octim
    
    @cached_property
    def enface_image(self):
        values = np.sum(self.image, axis=1)
        return (values - np.min(values)) / np.max(values) * 255
    
    def plot_rnfl_heatmap(self):
        if self.rnfl_mask is None:
            self.get_rnfl_mask()
        plot_enface(self.image)
        plot_enface_heatmap(self.rnfl_mask)
    
    def get_rnfl_mask(self):
        if self.rnfl_mask is None:
            rnfl_map = np.isin(self.mask, self.labels.RNFL).astype('uint8')
            vessels_map = np.isin(self.mask, self.labels.VESSELS).astype('uint8')
            background = np.isin(self.mask, self.labels.RNFL_AND_VESSELS).astype('uint8')

            for bscan in range(rnfl_map.shape[0]):
                background_filled = binary_fill_holes(background[bscan])
                dist_rnfl = distance_transform_edt(1-rnfl_map[bscan])
                dist_background = distance_transform_edt(background_filled)
                rnfl_map[bscan] = (binary_fill_holes(rnfl_map[bscan]) + (vessels_map[bscan] & (dist_rnfl < dist_background))) > 0
            self.rnfl_mask = rnfl_map
        return self.rnfl_mask
    
    def get_bmo_contour_and_mask(self, kernel_size=7):
        all_bm = np.isin(self.mask, self.labels.BM_LABELS).astype('uint8')

        self.all_bm = all_bm

        enface_bm = np.where(np.sum(all_bm, axis=1) > 0, 0, 1).astype('uint8')
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(kernel_size,kernel_size))
        try:
            closing = cv2.morphologyEx(enface_bm, cv2.MORPH_OPEN, kernel)
        except:
            raise ValueError('Closing failed while detecting BM opening')
        n_components, labels, stats, centroids = \
            cv2.connectedComponentsWithStats(
                closing, 1,  connectivity=4)
        margin_x = enface_bm.shape[0] / 4
        margin_y = enface_bm.shape[1] / 4
        is_in_center_mask_x = np.logical_and(
            (centroids[:, 1] > margin_x),
            ((enface_bm.shape[0] - centroids[:, 1]) > margin_x))
        is_in_center_mask_y = np.logical_and(
            (centroids[:, 0] > margin_y),
            ((enface_bm.shape[1] - centroids[:, 0]) > margin_y))
        is_in_center_mask = np.logical_and(
            is_in_center_mask_x, is_in_center_mask_y)
        sizes = stats[:, -1]
        sizes[~is_in_center_mask] = 0
        sizes_sorted = np.argsort(sizes)
        label = [sizes_sorted[-2]]

        bmo_mask = np.where(labels == label, 1, 0)\
            .astype('uint8')
        self.bmo_mask = bmo_mask

        # Fit ellipse
        contours, _ = cv2.findContours(
            bmo_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        cnt = contours[0].reshape(-1, 2)
        if len(contours[0]) < 5:
            raise ValueError('BMO contour too small')
        return contours, bmo_mask
        
    def detect_bm_opening(self, kernel_size=7):
        
        contours, mask = self.get_bmo_contour_and_mask(kernel_size)

        ellipse = cv2.fitEllipse(contours[0])

        self.bmo_center = ellipse[0]

        im_enface = np.sum(self.image, axis=1)
        im_enface = (im_enface - np.min(im_enface)) / np.max(im_enface)
        im_enface = im_enface / np.max(im_enface)

        pixel_area = self.res_depth_mm*self.res_width_mm
        # correct for diameter instead of radius
        area_ellipse = np.prod(ellipse[1])*np.pi*pixel_area/4

        area_selected = np.sum(mask)*pixel_area
        return {
            'BMO area projected (mm2)': area_selected,
            'BMO area projected fitted (mm2)': area_ellipse
        }
    
    def get_rnfl_thickness(self, rstart: float, rend: float):
        if not self.bmo_center:
            self.detect_bm_opening()

        grid = np.indices((self.mask.shape[0], self.mask.shape[2]))
        dist_z = np.abs(grid[0] - self.bmo_center[1]) * self.res_depth_mm
        dist_x = np.abs(grid[1] - self.bmo_center[0]) * self.res_width_mm
        dist_eucl = np.sqrt(dist_z**2 + dist_x**2)

        dist_mask = np.where((dist_eucl < rend) & (dist_eucl > rstart), 1, 0)
        rnfl_map = self.get_rnfl_mask()
        rnfl_thickness = np.sum(rnfl_map, axis=1)*self.res_height_mm
        thickness_measurements = rnfl_thickness[dist_mask > 0]

        return {
            f'RNFL thickness {rstart:.1f}-{rend:.1f}mm radius '
            f'mean (mm)': np.mean(thickness_measurements)
        }
    
    def get_ppa_area(self):
        ppa_alpha = np.where(np.sum(np.isin(self.mask, self.labels.PPA_A), axis=1) > 0, 1, 0)
        ppa_beta = np.where(np.sum(np.isin(self.mask, self.labels.PPA_B), axis=1) > 0, 1, 0)
        ppa_gamma = np.where(np.sum(np.isin(self.mask, self.labels.PPA_G), axis=1) > 0, 1, 0)

        return {
            'PPA - alpha area (mm2)':
            np.sum(ppa_alpha)
            * self.res_depth_mm
            * self.res_width_mm,
            'PPA - beta area (mm2)':
            np.sum(ppa_beta)
            * self.res_depth_mm
            * self.res_width_mm,
            'PPA - gamma area (mm2)':
            np.sum(ppa_gamma)
            * self.res_depth_mm
            * self.res_width_mm
        }
    
    def get_cup_depth(self):
        if not hasattr(self, 'bmo_mask'):
            self.detect_bm_opening()

        bm_skeleton = np.zeros(self.all_bm.shape)
        for i in range(len(self.all_bm)):
            bm_bscan = self.all_bm[i]
            if np.max(bm_bscan) > 0:
                bm_skeleton[i] = skeletonize(bm_bscan).astype(np.uint8)
        grid = np.indices(bm_skeleton.shape)

        contours, _ = cv2.findContours(
            self.bmo_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        cont_mask = np.zeros(
            (self.mask.shape[0], self.mask.shape[2]), dtype='uint8')
        cv2.drawContours(cont_mask, contours, 0, (1, 0), 10)

        bm_skeleton_edge_mask = np.repeat(
            cont_mask[:, np.newaxis, :], bm_skeleton.shape[1], axis=1)
        bm_skeleton_edge = bm_skeleton * bm_skeleton_edge_mask
        indices_bm = grid[:, bm_skeleton_edge == 1]
        z = indices_bm[0] * self.res_depth_mm  # bscans
        y = indices_bm[1] * self.res_height_mm  # height in bscan
        x = indices_bm[2] * self.res_width_mm  # width

        input_coordinates = np.array([x, z, np.ones(z.shape)]).T
        p, resid, rank, s = lstsq(input_coordinates, np.array([y]).T)
        p_x = p[0, 0]
        p_z = p[1, 0]
        y_inc = p[2, 0]

        plane_height = grid[0, :, 0]*self.res_depth_mm\
            * p_z + grid[2, :, 0] * p_x*self.resolution[2]\
            + np.ones(grid[1, :, 0].shape) * y_inc
        lc = np.isin(self.mask, [3]).astype(float)
        lc_skeleton = np.zeros(lc.shape)
        for i in range(len(lc)):
            lc_bscan = lc[i]
            if np.max(lc_bscan) > 0:
                lc_skeleton[i] = skeletonize(lc_bscan).astype(np.uint8)

        lc_points_pix = grid[:, lc_skeleton == 1]
        lc_points = (self.resolution * lc_points_pix.T).T
        distances = np.abs(
            p_x*lc_points[2] - lc_points[1] + p_z * lc_points[0] + y_inc)\
            / np.sqrt(p_x**2 + p_z**2 + 1)
        self.plane_height = plane_height

        is_below_bmo = grid[0, :, :]*self.res_depth_mm * p_z + grid[2, :, :]\
            * p_x*self.resolution[2] - grid[1, :, :]*self.resolution[1] + y_inc < 0

        rnfl_map = self.get_rnfl_mask()
        above_rnfl = np.cumsum(rnfl_map, axis=1) < 1
        above_rnfl_masked = above_rnfl * np.repeat(
            (np.sum(rnfl_map, axis=1, keepdims=True) > 0),
            above_rnfl.shape[1], axis=1)

        cup_mask = above_rnfl_masked * is_below_bmo
        bmo_mask_3d = np.repeat(
            np.expand_dims(self.bmo_mask, axis=1), above_rnfl.shape[1], axis=1)
        cup_mask_corrected = cup_mask * bmo_mask_3d
        cup_depth_corrected = np.sum(cup_mask_corrected, axis=1)
        volume = np.sum(cup_depth_corrected) * np.prod(self.resolution)

        if len(distances) == 0:
            return {'Cup volume below BMO (mm3)': volume}

        return {
            'LC mean distance to BMO plane (mm)': np.mean(distances),
            'LC median distance to BMO plane (mm)': np.median(distances),
            'Cup volume below BMO (mm3)': volume
        }

    def get_cp_circle(self, r:float, n = 400):
        angles = np.linspace(0, 2*np.pi, n)
        x = self.bmo_center[0] + (r / self.resolution[2]) * np.cos(angles)
        z = self.bmo_center[1] + (r / self.resolution[0]) * np.sin(angles)
        return x, z, angles

    def get_rnfl_thickness_circular(self, r: float):
        if not self.bmo_center:
            self.detect_bm_opening()
        rnfl_map = self.get_rnfl_mask()
        rnfl_thickness = np.sum(rnfl_map, axis=1)*self.resolution[1]

        x, z, angles = self.get_cp_circle(r)

        thickness_circular = rnfl_thickness[
            np.rint(z).astype(int), np.rint(x).astype(int)]
        results = {}

        clock_hours = self.get_clock_hours(angles, thickness_circular)
        for key, value in clock_hours.items():
            results[f'RNFL thickness d={2*r:.1f} mm hour {key} mean (mm)'] = np.mean(value)
            results[f'RNFL thickness d={2*r:.1f} mm hour {key} n measurements'] = len(value)
        quadrants = self.get_quadrants(angles, thickness_circular)
        for key, value in quadrants.items():
            results[f'RNFL thickness d={2*r:.1f} mm hour {key} mean (mm)'] = np.mean(value)
            results[f'RNFL thickness d={2*r:.1f} mm hour {key} n measurements'] = len(value)
        segments = self.get_tsnit_segments(angles, thickness_circular)
        for key, value in segments.items():
            results[f'RNFL thickness d={2*r:.1f} mm hour {key} mean (mm)'] = np.mean(value)
            results[f'RNFL thickness d={2*r:.1f} mm hour {key} n measurements'] = len(value)
        results[f'RNFL thickness d={2*r:.1f} mm global mean (mm)'] =\
            np.mean(thickness_circular)
        
        return results
    
    def get_radial_scan(self, angle):
        center_x = self.bmo_center[0]
        center_z = self.bmo_center[1]
        length_mm = 2.0
        distance_x = length_mm / self.resolution[2] *\
            np.sin(angle)
        distance_z = length_mm / self.resolution[0] *\
            np.cos(angle)
        
        startpoint_x = center_x - distance_x
        startpoint_z = center_z + distance_z
        endpoint_x = center_x + distance_x
        endpoint_z = center_z - distance_z

        x = np.linspace(startpoint_x, endpoint_x, 500)
        z = np.linspace(startpoint_z, endpoint_z, 500)
        segmentation = self.mask[np.rint(z).astype(int), :,
                                 np.rint(x).astype(int)].T
        if self.rnfl_mask is not None:
            rnfl_mask = self.rnfl_mask[np.rint(z).astype(int), :,
                                 np.rint(x).astype(int)].T
        else: rnfl_mask = None
        if self.image is not None:
            scan = self.image[np.rint(z).astype(int), :,
                            np.rint(x).astype(int)].T
        else:
            scan = np.zeros(segmentation.shape)
        res_x = 2*length_mm / 500
        bscan = OCTBScan(imagedata=scan, res_height_mm=self.res_height_mm, res_width_mm=res_x, laterality=self.laterality)
        return OpticDiscRadialBScan(bscan=bscan, mask=segmentation, rnfl_mask=rnfl_mask, 
                                    startpoint=(startpoint_x, startpoint_z), endpoint=(endpoint_x, endpoint_z),
                                    res_y_enface=self.resolution[0], res_x_enface=self.resolution[2])

    def get_radial_bscans(self):
        if not self.bmo_center:
            self.detect_bm_opening()
        angles = np.linspace(0, np.pi, 24, endpoint=False)
        bscans = [self.get_radial_scan(angle) for angle in angles]
        return bscans
    
    def plot_circular_positioning(self, regiontype='tsnit', r=1.5, ax=None):
        if ax is None:
            _, ax = plt.subplots()
        if not self.bmo_center:
            self.detect_bm_opening()

        x, y, angles = self.get_cp_circle(r)
        xvals = self.get_positioning(angles, x, regiontype)
        yvals = self.get_positioning(angles, y, regiontype)

        plot_image(self.enface_image, ax=ax)
        for k in xvals.keys():
            ax.scatter(xvals[k], yvals[k], label=k, s=1)
            ax.legend()
    
    def plot_enface_bmo(self, ax=None, kernel_size=7):
        if ax is None:
            _, ax = plt.subplots()
        contours, mask = self.get_bmo_contour_and_mask(kernel_size)
        im_2_show = self.enface_image.copy()
        ellipse = self.get_bmo_ellipse(contours[0])
        cv2.ellipse(im_2_show, ellipse, (0, 255, 0), 2)
        plot_image(im_2_show, ax=ax)
        contour_points = np.array([c[:,0] for c in contours])

        ax.scatter(contour_points[:, :, 0], contour_points[:, :, 1], c='r', s=1)

    def get_all_features(self):
        bmo_features = self.detect_bm_opening()
        if bmo_features is False:
            return {}
        rnfl_features = self.get_rnfl_thickness(1, 2.5)
        rnfl_features = rnfl_features | self.get_rnfl_thickness(1, 1.2)
        rnfl_features = rnfl_features | self.get_rnfl_thickness(1.5, 1.7)
        rnfl_features = rnfl_features | self.get_rnfl_thickness(2, 2.2)
        
        rnfl_features = rnfl_features | self.get_rnfl_thickness_circular(1.70)
        rnfl_features = rnfl_features | self.get_rnfl_thickness_circular(1.75)
        rnfl_features = rnfl_features | self.get_rnfl_thickness_circular(2.05)

        minimal_rim_width = self.get_minimal_rim_width()
        ppa_features = self.get_ppa_area()
        lc_features = self.get_cup_depth()
        all_features = (bmo_features | rnfl_features | ppa_features |
                        lc_features | minimal_rim_width)
        return all_features

@dataclass
class OpticDiscOCTEnfaceGroup(AbstractOpticDiscOCT):

    enface_image: np.ndarray = None
    bscans: List[OpticDiscBscan] = None
    laterality: str = None
    enface_resolution: float = None  

    def plot_circular_positioning(self, regiontype='tsnit', ax=None):
        if ax is None:
            _, ax = plt.subplots()
        x_dicts = []
        y_dicts = []
        for b in self.bscans:
            if isinstance(b, OpticDiscCircularBScan):
                xvals, yvals = b.get_circular_positioning(regiontype=regiontype)
                x_dicts.append(xvals)
                y_dicts.append(yvals)
        
        plot_image(self.enface_image, ax=ax)
        for k in xvals.keys():
            xvals = [d[k] for d in x_dicts]
            yvals = [d[k] for d in y_dicts]
            ax.scatter(np.concatenate(xvals), np.concatenate(yvals), label=k, s=1)
            ax.legend()

    def get_all_features(self):
        results = {}
        for b in self.bscans:
            if isinstance(b, OpticDiscCircularBScan):
                results = results | b.get_all_features()
        results = results | self.get_minimal_rim_width()
        return results

    def get_radial_bscans(self):
        return [bscan for bscan in self.bscans if isinstance(bscan, OpticDiscRadialBScan)]

    def get_bmo_contour_points(self):
        contour_points_0 = []
        contour_points_1 = []
        for i, bscan in enumerate(self.bscans):
            if not isinstance(bscan, OpticDiscRadialBScan):
                continue
            if not bscan.bmo_center:
                bscan.detect_bm_opening()
            start_bscan = bscan.bmo_start_x
            end_bscan = bscan.bmo_end_x
            x_locations = np.linspace(bscan.startpoint[1], bscan.endpoint[1], bscan.image.shape[1])
            y_locations = np.linspace(bscan.startpoint[0], bscan.endpoint[0], bscan.image.shape[1])
            start_x = x_locations[start_bscan]
            start_y = y_locations[start_bscan]
            end_x = x_locations[end_bscan]
            end_y = y_locations[end_bscan]
            contour_points_0.append([(start_y, start_x)])
            contour_points_1.append([(end_y, end_x)])
        contour_points = np.concatenate([contour_points_0, contour_points_1])
        contour_points = np.array(contour_points).astype(np.float32)
        return contour_points

    def get_bmo_area(self):
        contour_points = self.get_bmo_contour_points()
        ellipse = self.get_bmo_ellipse(contour_points)    
        # correct for diameter instead of radius
        pixel_area = self.enface_resolution**2
        area_ellipse = np.prod(ellipse[1])*np.pi*pixel_area/4
        return {
            'BMO area projected fitted (mm2)': area_ellipse
        }
    
    def plot_bmo_contour(self, ax=None):
        if ax is None:
            _, ax = plt.subplots()
        contour_points = self.get_bmo_contour_points()
        im_2_show = self.enface_image.copy()
        ellipse = self.get_bmo_ellipse(contour_points)
        cv2.ellipse(im_2_show, ellipse, (0, 255, 0), 5)
        plot_image(im_2_show, ax=ax)
        contour_points = np.array([c[0] for c in contour_points])

        ax.scatter(contour_points[:, 0], contour_points[:, 1], c='r', s=1)


