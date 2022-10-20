from convpoint.nn import PtConv
from convpoint.nn.utils import apply_bn
import torch
import torch.nn as nn
import torch.nn.functional as F

#
# PointNet module to determine roof face characteristics (e.g. slope, face presence, etc.)
#

class RoofN3DNetPointNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(RoofN3DNetPointNet, self).__init__()
        
        self.conv1 = nn.Conv1d(in_channels, 64, 1)
        self.bnor1 = nn.BatchNorm1d(64)
        
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.bnor2 = nn.BatchNorm1d(128)
        
        self.conv3 = nn.Conv1d(128, 256, 1)
        self.bnor3 = nn.BatchNorm1d(256)
        
        self.full1 = nn.Linear(256, 128)
        # batch norm layer?
        
        self.full2 = nn.Linear(128, 64)
        # batch norm layer?
        
        self.full3 = nn.Linear(64, out_channels)
                
    def forward(self, input):
                
        out = F.relu(self.bnor1(self.conv1(input)))
        out = F.relu(self.bnor2(self.conv2(out)))
        out = F.relu(self.bnor3(self.conv3(out)))
        
        # avg/mean pooling improves robustness of training and 
        # better predictions
        #out, _ = torch.max(out, 2, keepdim=True)
        out = torch.mean(out, 2, keepdim=True)
                
        out = out.view(1, -1)

        out = self.full1(out)
        # batch norm layer?
        out = F.relu(out)
        
        out = self.full2(out)
        # batch norm layer?
        out = F.relu(out)
        
        out = self.full3(out)

        return out
        
#
# Main RoofN3DNet model with ConvPoint U-net feature extraction backbone
#
        
class RoofN3DNet(nn.Module):
    def __init__(self, in_channels, out_channels, roof_faces, dimension=3, args={}):
        super(RoofN3DNet, self).__init__()
        
        n_centers = 16

        pl = 48
        
        # ConvPoint feature extraction
        self.cv1 = PtConv(in_channels, 2*pl, n_centers, dimension, use_bias=False)
        self.cv2 = PtConv(2*pl, 2*pl, n_centers, dimension, use_bias=False)
        self.cv3 = PtConv(2*pl, 2*pl, n_centers, dimension, use_bias=False)

        self.cv3d = PtConv(2*pl, 2*pl, n_centers, dimension, use_bias=False)
        self.cv2d = PtConv(4*pl, 2*pl, n_centers, dimension, use_bias=False)
        self.cv1d = PtConv(4*pl, 2*pl, n_centers, dimension, use_bias=False)

        # point-wise classification (semantic/part segmentation)
        self.fcout = nn.Linear(2*pl, roof_faces+1)

        self.bn1 = nn.BatchNorm1d(2*pl)
        self.bn2 = nn.BatchNorm1d(2*pl)
        self.bn3 = nn.BatchNorm1d(2*pl)

        self.bn3d = nn.BatchNorm1d(2*pl)
        self.bn2d = nn.BatchNorm1d(2*pl)
        self.bn1d = nn.BatchNorm1d(2*pl)

        self.drop = nn.Dropout(0.5)
                            
        # roof face characteristics with PointNet layers 
        # in_channels: 2*pl (96) feature channels + 3 for concatenated xyz = 99
        # out_channels: e.g. 18 slope classes (5 degree intervals) + 1 slope residual + 1 objectness = 20
        self.point_nets = nn.ModuleList([RoofN3DNetPointNet(in_channels=2*pl+3, out_channels=out_channels) for i in range(1, roof_faces+1)])
    
    def forward(self, x, input_pts, return_features=False):
        # x: [1, N, 1], input_pts: [1, N, 3]

        # ConvPoint feature extraction
                
        x1, pts1 = self.cv1(x, input_pts, 8, 64)
        x1 = F.relu(apply_bn(x1, self.bn1)) # -> [1, 64, 2*pl]

        x2, pts2 = self.cv2(x1, pts1, 8, 16)
        x2 = F.relu(apply_bn(x2, self.bn2)) # -> [1, 16, 2*pl]

        x3, pts3 = self.cv3(x2, pts2, 4, 8)
        x3 = F.relu(apply_bn(x3, self.bn3)) # -> [1, 8, 2*pl]

        x3d, _ = self.cv3d(x3, pts3, 4, pts2)
        x3d = F.relu(apply_bn(x3d, self.bn3d)) # -> [1, 16, 2*pl]
        x3d = torch.cat([x3d, x2], dim=2) # -> [1, 16, 4*pl]
        
        x2d, _ = self.cv2d(x3d, pts2, 4, pts1)
        x2d = F.relu(apply_bn(x2d, self.bn2d)) # -> [1, 64, 2*pl]
        x2d = torch.cat([x2d, x1], dim=2) # -> [1, 64, 4*pl]
        
        x1d, _ = self.cv1d(x2d, pts1, 8, input_pts)
        x1d = F.relu(apply_bn(x1d, self.bn1d)) # -> [1, N, 2*pl]
        # no concat, since there are no input features
        
        # point-wise classification (semantic / part segmentation)
        
        xout = x1d
        xout = xout.view(-1, xout.size(2))
        xout = self.drop(xout)
        xout = self.fcout(xout)
        xout = xout.view(x.size(0), -1, xout.size(1))

        # roof face characteristics with PointNet layers 
        
        # determine class index (with highest score) per point (as in semantic segmentation) [1, N, 5] -> [1, N]
        cls_idx_per_pt = torch.argmax(xout, dim=2) 
                
        # prepare roof face features for PointNet modules (ConvPoint features + xyz)
        rf_fts = torch.cat([x1d, input_pts], dim=2) 
        
        # list for roof face outputs
        out_roof_faces = []
        
        for c, pnet in enumerate(self.point_nets, 1):
            
            # extract roof face features for points of class c
            rf_fts_c = rf_fts[:, cls_idx_per_pt.view(-1).eq(c), :]
        
            # sample fixed number of points (here 32) for this class [1, N, 99] -> [1, 32, 99]
            # or construct a tensor with zeros of the same shape
            if rf_fts_c.shape[1] != 0:
                rf_fts_c_32 = rf_fts_c[:, torch.randint(0, rf_fts_c.shape[1], (32,)), :]
            else:
                rf_fts_c_32 = torch.zeros(1, 32, 99).cuda()

            # append the output of the PointNet module to roof face output list
            # PointNet module expects the tensor axis 1 and 2 swapped, e.g. [1, 32, 99] -> [1, 99, 32]
            out_roof_faces.append(pnet(torch.transpose(rf_fts_c_32, 1, 2)))

        # return point segmentation & (stacked) roof face characteristics
        return xout, torch.stack(out_roof_faces, dim=1)