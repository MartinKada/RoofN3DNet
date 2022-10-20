from scipy.spatial import HalfspaceIntersection
from scipy.spatial import ConvexHull

from RoofN3DDataset import roof_surface_label
import numpy as np
from shapely import wkt
import math

# Constructs a triangle mesh from a set of halfspaces. The feasible point
# must be located behind all halfspaces (inside the convex solid).
def construct_mesh_from_halfspaces(halfspaces, feasible_point):
    
    hsi = HalfspaceIntersection(np.array(halfspaces), np.array(feasible_point))
                
    hull = ConvexHull(hsi.intersections)        
    
    # ConvexHull does not provide all simplices in the correct orientation.
    # It is therefore necessary to flip some simplices (triangles), so that 
    # the feasible point is always on the negative (back) side of the triangle
    
    simplices = hull.simplices.copy()

    for i, s in enumerate(hull.simplices):
        v01 = hull.points[s[1]] - hull.points[s[0]]
        v02 = hull.points[s[2]] - hull.points[s[0]]
        x = v01[1] * v02[2] - v01[2] * v02[1]
        y = v01[2] * v02[0] - v01[0] * v02[2]
        z = v01[0] * v02[1] - v01[1] * v02[0]
        l = math.sqrt(x*x+y*y+z*z)
        A = x / l
        B = y / l
        C = z / l
        D = -(A*hull.points[s[0]][0]+B*hull.points[s[0]][1]+C*hull.points[s[0]][2])

        if (A * feasible_point[0] + B * feasible_point[1] + C * feasible_point[2] + D) < 0:
            simplices[i] = np.flip(simplices[i])   
            
    return hull.points.copy(), simplices.copy()

# RoofN3D does not provide any rectangular bounding boxes of the buildings. The function
# therefore creates vectors that are parallel to such a bounding box, and that can later
# be used to rotate an upwards facing plane to get the planes of the roof faces. For this
# purpose, the first surface (roof face) of the ground truth data is used to aligh the
# bounding box with the roof faces (and therefore with the ridge line).
def determine_side_axes(surfaces):

    rot_axes = np.zeros((4, 3))
    
    # There should be at least 2 roof (sur)faces per building. Or, if we assume to also have 
    # shed roofs with 1 roof face, then we can use the first roof (sur)face to align the
    # bounding box with.
    # Create (normalized) vector r that is perpendicular to the normal vector of the first 
    # (sur)face and the upward vector (by using the cross product).
    surface = surfaces.iloc[0]
    plane = [surface['plane_a'], surface['plane_b'], surface['plane_c'], surface['plane_d']]
    rx = plane[1] * 1.0 - plane[2] * 0.0
    ry = plane[2] * 0.0 - plane[0] * 1.0
    rz = plane[0] * 0.0 - plane[1] * 0.0
    length = math.sqrt(rx*rx+ry*ry+rz*rz)
    rx = rx / length
    ry = ry / length
    rz = rz / length

    # Depending on the orientation of the first (sur)face, the just computed vector is rotated
    # to get the rotation axis of the roof face with label 1 (at index 0).
    l = roof_surface_label(plane[0], plane[1])

    if l == 1:
        rot_axes[0] = (rx, ry, rz)
    elif l == 2:
        rot_axes[0] = (ry, -rx, rz)
    elif l == 3:
        rot_axes[0] = (-rx, -ry, rz)
    elif l == 4:
        rot_axes[0] = (-ry, rx, rz)
            
    # Create the other rotation axes from first one.
    rot_axes[1] = (-rot_axes[0][1],  rot_axes[0][0], rot_axes[0][2])
    rot_axes[2] = (-rot_axes[0][0], -rot_axes[0][1], rot_axes[0][2])
    rot_axes[3] = ( rot_axes[0][1], -rot_axes[0][0], rot_axes[0][2])        
        
    return rot_axes

# Construct the halfspaces of the four sides that serve as the oriented bounding box of the
# input point cloud. For this, the rotation axes are used to first create the normal vector
# of the plane equation, and then the maximum of all dot products of the input points with 
# this normal vector determines the last (D) parameter of the plane equation, which is also
# the equation of the halfspace.
def construct_side_halfspaces(rot_axes, points):
    
    halfspaces = []

    # create plane equation from each rotation axis
    for axes in rot_axes:
        plane = np.array([axes[1], -axes[0], 0.0])
        D = np.max(np.dot(points, plane))
        
        halfspaces.append([plane[0], plane[1], plane[2], -D])
        
    return halfspaces

# Construct roof halfspaces by rotating a plane with upwards normal vector (0,0,1) by the 
# rotation axes of the roof faces by the predicted angles. The location (D parameter) of
# the plane equation of the halfspace is then determined from the mean position of all
# points that are predicted to belong to this roof face. Only those halfspaces are 
# constructed for which the roof faces are predicted to exist (objectness == 1).
def construct_roof_halfspaces(rot_axes, points, labels, angles, objectness, num_faces=4):

    # for each roof face, determine the number of roof points, and if there
    # any, then also compute the mean point coordinates
    number_of_face_points = np.zeros((num_faces))
    
    mean_roof_face_points = np.zeros((num_faces,3))
    
    for i in range(num_faces):
        face_points = points[labels==i+1]
        number_of_face_points[i] = len(face_points)
        if number_of_face_points[i] > 0:
            mean_roof_face_points[i] = np.mean(face_points, axis=0)

    halfspaces = []
            
    # create for all faces the halfspace as described above
    for i in range(num_faces):
        
        if objectness[i] < 0.0:
            continue
        
        if number_of_face_points[i] == 0:
            continue
        
        rx = rot_axes[i][0]
        ry = rot_axes[i][1]
        rz = rot_axes[i][2]
        
        angle = angles[i]
        
        angle = math.radians(angle)
                
        # define a quaternion for rotation around axis by angle
        angle2 = angle / 2.0;
        r0 = math.cos(angle2);
        sin_angle = math.sin(angle2)
        r1 = rx * sin_angle
        r2 = ry * sin_angle
        r3 = rz * sin_angle

        vX = 0.0
        vY = 0.0
        vZ = 1.0
        
        # perform rotation of vector (0,0,1)
        A = (2.0 * r0 * r0 - 1.0 + 2.0 * r1 * r1) * vX + \
            (2.0 * r1 * r2 + 2.0 * r0 * r3) * vY + \
            (2.0 * r1 * r3 - 2.0 * r0 * r2) * vZ
            
        B = (2.0 * r1 * r2 - 2.0 * r0 * r3) * vX + \
            (2.0 * r0 * r0 - 1.0 + 2.0 * r2 * r2) * vY + \
            (2.0 * r2 * r3 + 2.0 * r0 * r1) * vZ
            
        C = (2.0 * r1 * r3 + 2.0 * r0 * r2) * vX + \
            (2.0 * r2 * r3 - 2.0 * r0 * r1) * vY + \
            (2.0 * r0 * r0 - 1.0 + 2.0 * r3 * r3) * vZ
        
        # create halfspace equation and store halfspace
        plane = [A, B, C, 0.0]
        
        m = mean_roof_face_points[i]
        D = -(plane[0] * m[0] + plane[1] * m[1] + plane[2] * m[2])

        plane[3] = D
        
        halfspaces.append(plane)

    return halfspaces

# Construct a 3D building model by halfspace modeling, and return the points
# and triangles (3 indices to the points) of the resulting triangle mesh.
def construct_3Dmodel(item, lbs, slope_angles, objectness):
    
    building = item["building"]
    
    building_pts = np.array([[p.x, p.y, p.z] for p in wkt.loads(building.points)])
    
    building_pts_center = 0.5 * (np.min(building_pts, axis=0) + np.max(building_pts, axis=0))

    bb_min = np.min(building_pts, axis=0) - building_pts_center
    bb_max = np.max(building_pts, axis=0) - building_pts_center
    
    # enforce minimum height of 2.0m for bottom halfspace
    bottom_z = bb_min[2]
    if bb_max[2] - bb_min[2] < 2.0:
        bottom_z = bb_max[2] - 2.0
    
    halfspaces = [[0.0, 0.0, -1.0, bottom_z]]
        
    # locate feasible point needed for intersection at 0.05m above bottom halfspace
    # !!!! Changing this value (0.05) to a lower number sometimes helps if the model construction fails.
    feasible_point = [0.0, 0.0, bottom_z + 0.05]
        
    # add halfspaces of the 4 facade sides
    rot_axes = determine_side_axes(item["surfaces"])
    halfspaces += construct_side_halfspaces(rot_axes, building_pts - building_pts_center)

    # add halfspaces of the roof faces
    halfspaces += construct_roof_halfspaces(rot_axes, building_pts - building_pts_center, lbs, slope_angles, objectness, num_faces=4)    
    
    # contruct mesh model by halfspace intersection
    model_pts, model_triangles = construct_mesh_from_halfspaces(halfspaces, feasible_point)
        
    return model_pts, model_triangles