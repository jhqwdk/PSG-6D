import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from vnn_net.vnn import MY_VNN_SimplePointnet_onlyp
from utils.rotation_utils import Ortho6d2Mat
from modules import ModifiedResnet, PointNet2MSG
from losses import SmoothL1Dis, ChamferDis, PoseDis
from model.Golbal_enhance import global_enhancev2


class PoseNetGT(nn.Module):
    def __init__(self, nclass=6, nprior=1024):
        super(PoseNetGT, self).__init__()
        self.nclass = nclass
        self.nprior = nprior

        self.rgb_extractor = ModifiedResnet()
        self.pts_extractor = PointNet2MSG(radii_list=[[0.01, 0.02], [0.02,0.04], [0.04,0.08], [0.08,0.16]])
        self.pts_gt_extractor = PointNet2MSG(radii_list=[[0.05,0.10], [0.10,0.20], [0.20,0.30], [0.30,0.40]])
        self.pose_estimator_aux = HeavyEstimator()

        self.glo_enhance1 = global_enhancev2(128)
        self.glo_enhance2 = global_enhancev2(128)

    def forward(self, inputs):
        end_points = {}

        rgb = inputs['rgb']
        pts = inputs['pts']
        choose = inputs['choose']
        pts_w_gt = inputs['qo']

        c = torch.mean(pts, 1, keepdim=True)
        pts = pts - c
        b = pts.size(0)

        # rgb feat
        rgb_local = self.rgb_extractor(rgb) # extract image level feature
        d = rgb_local.size(1)
        rgb_local = rgb_local.view(b, d, -1)
        choose = choose.unsqueeze(1).repeat(1, d, 1)
        rgb_local = torch.gather(rgb_local, 2, choose).contiguous()

        pts_local = self.pts_extractor(pts)
        pts_local = pts_local.permute(0,2,1) #[12,1024,128]
        pts_local = self.glo_enhance1(pts_local)
        pts_local = pts_local.permute(0,2,1) #[12,128,1024]

        pts_local_w_gt = self.pts_gt_extractor(pts_w_gt)
        pts_local_w_gt = pts_local_w_gt.permute(0,2,1) #[12,1024,128]
        pts_local_w_gt = self.glo_enhance2(pts_local_w_gt)
        pts_local_w_gt = pts_local_w_gt.permute(0,2,1) #[12,128,1024]
        r_aux_gt, t_aux_gt, s_aux_gt = self.pose_estimator_aux(pts, pts_w_gt, rgb_local.detach(), pts_local.detach(), pts_local_w_gt)
    

        end_points["pts_local_w_gt"] = pts_local_w_gt
        end_points['pred_rotation'] = r_aux_gt
        end_points['pred_translation'] = t_aux_gt + c.squeeze(1)
        end_points['pred_size'] = s_aux_gt

        return end_points

class SupervisedLoss(nn.Module):
    def __init__(self, cfg):
        super(SupervisedLoss, self).__init__()
        self.cfg = cfg.loss

    def forward(self, end_points):
        t = end_points['pred_translation']
        r = end_points['pred_rotation']
        s = end_points['pred_size']
        loss = self._get_loss(r, t, s, end_points)
        return loss
  
    def _get_loss(self, r, t, s, end_points):
        loss = PoseDis(r, t, s, end_points['rotation_label'],end_points['translation_label'],end_points['size_label'])
        return loss



class HeavyEstimator(nn.Module):
    def __init__(self):
        super(HeavyEstimator, self).__init__()

        self.pts_mlp1 = nn.Sequential(
            nn.Conv1d(3, 32, 1),
            nn.ReLU(),
            nn.Conv1d(32, 64, 1),
            nn.ReLU()
        )

        self.pts_so_mlp1 = nn.Sequential(
            nn.Conv1d(192, 384, 1),
            nn.ReLU(),
            nn.Conv1d(384, 64, 1),
            nn.ReLU(),
        )

        self.pts_mlp2 = nn.Sequential(
            nn.Conv1d(3, 32, 1),
            nn.ReLU(),
            nn.Conv1d(32, 64, 1),
            nn.ReLU()
        )
        self.pose_mlp1 = nn.Sequential(
            nn.Conv1d(64+64+384+64, 256, 1),
            nn.ReLU(),
            nn.Conv1d(256, 256, 1),
            nn.ReLU(),
        )
        self.pose_mlp2 = nn.Sequential(
            nn.Conv1d(512, 512, 1),
            nn.ReLU(),
            nn.Conv1d(512, 512, 1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.rotation_estimator = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 6),
        )
        self.translation_estimator = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 3),
        )
        self.size_estimator = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 3),
        )

        self.vnn = MY_VNN_SimplePointnet_onlyp(c_dim=64, hidden_dim=128,meta_output='equivariant_latent')

    def forward(self, pts, pts_w, rgb_local, pts_local, pts_w_local):
        pts_so=pts
        pts = self.pts_mlp1(pts.transpose(1,2))
        pts_w = self.pts_mlp2(pts_w.transpose(1,2))

        B=pts_so.size(0)
        x_list = np.array([], dtype=np.float32)
        device = pts_so.device
        x_list = torch.tensor(x_list, device=device)
        for i in range(B):  # ([6, N])
            x_i = pts_so[i].unsqueeze(1)  # [6, 1, N]
            x_i = x_i.transpose(0, 1)  #[1024,1,3]->[1,1024,3]
            x_i = self.vnn(x_i).unsqueeze(0)  #[1,192,1024]
            x_list = torch.cat((x_list, x_i), dim=0)
        
        pts_so = x_list  # [12, 192, 1024])
        pts_so = self.pts_so_mlp1(pts_so)


        pose_feat = torch.cat([rgb_local, pts, pts_local, pts_w, pts_w_local,pts_so], dim=1)

        pose_feat = self.pose_mlp1(pose_feat)
        pose_global = torch.mean(pose_feat, 2, keepdim=True)
        pose_feat = torch.cat([pose_feat, pose_global.expand_as(pose_feat)], 1)
        pose_feat = self.pose_mlp2(pose_feat).squeeze(2)
        r = self.rotation_estimator(pose_feat)
        r = Ortho6d2Mat(r[:, :3].contiguous(), r[:, 3:].contiguous()).view(-1,3,3)
        t = self.translation_estimator(pose_feat)
        s = self.size_estimator(pose_feat)
        return r,t,s
