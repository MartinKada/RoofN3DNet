# add the parent folder to the python path to access convpoint library
import sys
sys.path.append('../')

import os
from pathlib import Path

import numpy as np
import random

import pandas as pd
from shapely import wkt
import e13tools
from sklearn.model_selection import train_test_split

import torch

def roof_surface_label(x, y):
    if x > 0.0 and y > 0.0:
        return 1
    elif x < 0.0 and y > 0.0:
        return 2
    elif x < 0.0 and y < 0.0:
        return 3
    elif x > 0.0 and y < 0.0:
        return 4
    else:
        print("Warning in roof_surface_label() function!")

def slope_class_from_z_value(z):
    deg = np.degrees(np.arccos(z))
    return (deg / 5.0).astype(int)        

def slope_class_and_residual_from_z_value(z):
    deg = np.degrees(np.arccos(z))
    cls = (deg / 5.0).astype(int)
    res = deg - (cls*5.0+2.5)
    return deg, cls, res
       
def angles_from_slopes_residuals(slopes, residuals):
    return torch.argmax(slopes, dim=-1, keepdim=True) * 5.0 + 2.5 + residuals
    # REPLACE WITH NUMPY ARGMAX WITH KEEPDIMS WHEN UPDATING TO NEWER NUMPY VERSION
    
class RoofN3DDataset():
    
    def __init__ (self, split=[0.85, 0.10, 0.05], training=True, return_building_and_surfaces=False):

        self.training = training
        self.return_building_and_surfaces = return_building_and_surfaces
        
        # Load building file
        fp_buildings = os.path.join("../", "RoofN3D/roofn3d_buildings.csv")
        print("Loading", fp_buildings)
        self.df_buildings = pd.read_csv(fp_buildings)
        
        # Load surface growing file with surface face segments
        fp_surfaces = os.path.join("../", "RoofN3D/roofn3d_surfacegrowing.csv")
        print("Loading", fp_surfaces)
        self.df_surfaces = pd.read_csv(fp_surfaces)
                
        print("RoofN3D dataset loaded with", len(self.df_buildings), "buildings.")
            
        print("Filtering buildings - keeping buildings with number of points in range (100 <= N <= 350).")  
        building_points = self.df_buildings.points.apply(wkt.loads)
        building_num_points = building_points.apply(len).values
        self.df_buildings = self.df_buildings[np.logical_and(building_num_points >= 100, building_num_points <= 350)]
        
        print("RoofN3D dataset now with", len(self.df_buildings), "buildings.")
        
        # create an array that denotes the different building classes
        building_classes = self.df_buildings['class'].copy()
        building_classes[building_classes == 'Saddleback roof'] = 0
        building_classes[building_classes == 'Two-sided hip roof'] = 1
        building_classes[building_classes == 'Pyramid roof'] = 2
        building_classes = building_classes.to_numpy()
        
        # create train, validation, test split by indices
        idx_all = np.arange(0, len(self.df_buildings))

        if split[0] == 0.0 or sum(split) != 1.0:
            print("Warning. Invalid split specified. Using all data for training.")
            self.idx_train = idx_all
            self.idx_validation = []
            self.idx_test = []
        elif split[1] == 0.0 and split[2] == 0.0:
            print("Warning. Test.")
            self.idx_train = idx_all
            self.idx_validation = []
            self.idx_test = []
        else:
            self.idx_train, idx_rest = train_test_split(idx_all, train_size=split[0], random_state=7, stratify=building_classes)
    
            if split[1] == 0.0:
                self.idx_validation = []
                self.idx_test = idx_rest
            elif split[2] == 0.0:
                self.idx_validation = idx_rest
                self.idx_test = []
            else:
                self.idx_validation, self.idx_test = train_test_split(idx_rest, train_size=split[1]/(split[1] + split[2]), random_state=12, stratify=building_classes[idx_rest])  
                
        self.idx = self.idx_train
                
        print(f"Using {len(self.idx_train)} buildings for training, {len(self.idx_validation)} for validation, and {len(self.idx_test)} for testing.")
        
    def __getitem__(self, index):
                                      
        # get building at given index
        building = self.df_buildings.iloc[self.idx[index]]
                
        # convert building points into NumPy array
        building_points = np.array([[p.x, p.y, p.z] for p in wkt.loads(building.points)])
        
        # construct labels array with same number of rows as numer of points
        point_labels = np.zeros((building_points.shape[0]), dtype=int)            

        # query the surfaces related to the building
        surfaces = self.df_surfaces.query(f'fk_buildings == {building.id}')
        
        # make a copy, as building points are still used to determine which building points are in which surface
        pts = building_points.copy()
                                    
        # translate points, so that the center of the bounding box lies in the origin of the coordinate system
        min = np.min(pts, axis=0)
        max = np.max(pts, axis=0)
        pts -= 0.5 * (min + max)
        
        # random rotation of points around point cloud center for training
        if self.training:
            rotation_angle = np.random.uniform() * 2 * np.pi
            cosval = np.cos(rotation_angle)
            sinval = np.sin(rotation_angle)
            rotation_matrix = np.array([[cosval, sinval, 0],
                                        [-sinval, cosval, 0],
                                        [0, 0, 1],])            
            pts = np.dot(pts, rotation_matrix)
        
        # determine roof surface segmentation point labels
        for index, row in surfaces.iterrows():
            surface_points = np.array([[p.x, p.y, p.z] for p in wkt.loads(row['points'])])
            plane = np.array([row['plane_a'], row['plane_b'], row['plane_c']])
            if self.training:
                plane = np.dot(plane, rotation_matrix)
            point_labels[e13tools.numpy.isin(building_points, surface_points)] = roof_surface_label(plane[0], plane[1])        

        # determine roof face characteristics
        face_slope_classes = np.zeros((4,1), dtype=np.int64)
        face_slope_residuals = np.zeros((4,1), dtype=np.float32)
        face_slope_angles = np.zeros((4,1), dtype=np.float32)
        face_objectness = np.zeros((4,1), dtype=np.int64)
               
        for index, row in surfaces.iterrows():
            plane = np.array([row['plane_a'], row['plane_b'], row['plane_c']])
            if self.training:
                plane = np.dot(plane, rotation_matrix)
            l = roof_surface_label(plane[0], plane[1])
            face_slope_angles[l-1][0], face_slope_classes[l-1][0], face_slope_residuals[l-1][0] = slope_class_and_residual_from_z_value(plane[2])
            face_objectness[l-1][0] = 1
                
        # return as dictionary
        ret_dict = {
            "points" : pts.astype(np.float32),
            "features" : np.ones((pts.shape[0], 1), dtype=np.float32),
            "points_labels" : point_labels.astype(np.int64),
            "face_slope_classes" : face_slope_classes,
            "face_slope_residuals" : face_slope_residuals,
            "face_slope_angles" : face_slope_angles,
            "face_objectness" : face_objectness,
        }
        
        if self.return_building_and_surfaces:
            ret_dict["building"] = building.copy()
            ret_dict["surfaces"] = self.df_surfaces.query(f'fk_buildings == {building.id}').copy()
                
        return ret_dict
                         
    def __len__(self):
        return len(self.idx)
    
    def set_training(self):
        self.training = True
        self.idx = self.idx_train

    def set_validation(self):
        self.training = False
        self.idx = self.idx_validation
    
    def set_testing(self):
        self.training = False
        self.idx = self.idx_test

