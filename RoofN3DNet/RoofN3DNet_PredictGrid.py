import sys
import torch
import os

import argparse
from datetime import datetime

sys.path.append('../')

from RoofN3DNet import RoofN3DNet
from RoofN3DDataset import RoofN3DDataset
from RoofN3DDataset import angles_from_slopes_residuals
from RoofN3DModel import construct_3Dmodel

from plyfile import PlyElement, PlyData
import numpy as np
from shapely import wkt
from tqdm import tqdm

def predict_roof_parameters(net, item, num_faces, num_slope_classes):

    with torch.no_grad():

        pts = item["points"] # [N, 3]

        # make prediction with data
        pr_seg, pr_roof_faces = net(
            torch.from_numpy(item["features"]).cuda()[None,:,:], 
            torch.from_numpy(pts).cuda()[None,:,:])

        pr_slope_cls, pr_slope_res, pr_face_obj = torch.split(pr_roof_faces, (num_slope_classes, 1, 1), dim=2)          

        # create point array with predicted labels (x,y,z,l)
        lbs = pr_seg.cpu().numpy().reshape((-1, num_faces)).argmax(1)
        
        pr_slope_angles = angles_from_slopes_residuals(pr_slope_cls, pr_slope_res)
        
        return lbs, pr_slope_angles, pr_face_obj

    
def reconstruct_3D_building(net, item, num_faces, num_slope_classes):
    
    lbs, pr_slope_angles, pr_face_obj = predict_roof_parameters(net, item, num_faces+1, num_slope_classes)

    slope_angles_np = torch.reshape(pr_slope_angles, (4, 1)).cpu().detach().numpy()
    pr_face_obj_np = torch.reshape(pr_face_obj, (4, 1)).cpu().detach().numpy()    
    
    model_pts, model_triangles = construct_3Dmodel(item, lbs, slope_angles_np, pr_face_obj_np)

    return lbs, model_pts, model_triangles
   

def save_as_PLY_file(filename, points, triangles):
    
    vertices = np.array([tuple(i) for i in points], dtype=[('x', 'f8'), ('y', 'f8'), ('z', 'f8')])

    faces = np.array([(tuple(i),) for i in triangles] , dtype=[('vertex_indices', 'i4', (3,))])

    vertex_element = PlyElement.describe(vertices, 'vertex')
    face_element = PlyElement.describe(faces, 'face')
    plydata = PlyData([vertex_element, face_element])
    
    plydata.write(filename)
    
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--savedir", type=str, default="./Saves")
    args = parser.parse_args()

    # construct network and load weights
    print("Creating RoofN3DNet network and loading weights...", end="", flush=True)
    net = RoofN3DNet(in_channels=1, out_channels=20, roof_faces=4)
    net.load_state_dict(torch.load(os.path.join(args.savedir, "state_dict.pth")))
    net.cuda()
    print("Done")

    #
    print("Loading RoofN3DDataset...")
    dataset = RoofN3DDataset(return_building_and_surfaces=True)
    dataset.set_testing()
    
    num_faces = 4
    num_slope_classes = 18

    spacing = 20.0

    np.random.seed(10)

    time_string = datetime.now().strftime('%Y-%m-%d-%H-%M')    
    
    idx_grid = np.random.randint(len(dataset), size=(25, 25))

    all_vertices = []
    all_faces = []
    all_points = []

    count_points = 0

    it = np.nditer(idx_grid, flags=['multi_index'])

    t = tqdm(it, ncols=200, desc="Constructing")

    for idx in t:

        lbs, model_pts, model_triangles = reconstruct_3D_building(net, dataset[idx], num_faces, num_slope_classes)

        # translate points by spacing (according to multi index grid), concatenate with predicted labels,
        # and append to list of point clouds (that are later stacked into one final point cloud)
        all_points.append(np.concatenate(
            (dataset[idx]["points"] + [it.multi_index[0]*spacing, it.multi_index[1]*spacing, 0.0], 
             lbs[:,np.newaxis]), axis=1))    

        all_vertices.append(model_pts + [it.multi_index[0]*spacing, it.multi_index[1]*spacing, 0.0])

        # add the number of points already in the vertex list to the indices of the triangles
        all_faces.append(model_triangles + count_points)
        count_points += model_pts.shape[0]

        t.set_postfix(ordered_dict = {"BUILDING":0}, BUILDING=(f"{idx}"))

    # write predicted point segmentation
    np.savetxt(os.path.join("../Results/", time_string + f"-RoofN3D-PR_Points.txt"), np.vstack(all_points), fmt=['%.4f','%.4f','%.4f', '%i'])

    # write constructed models as PLY triangle meshes
    save_as_PLY_file(os.path.join("../Results/", time_string + f"-RoofN3D-PR_Buildings.ply"), np.vstack(all_vertices), np.vstack(all_faces))
 

if __name__ == '__main__':
    main()
    print('{}-Done.'.format(datetime.now()))