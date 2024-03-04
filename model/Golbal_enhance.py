import torch
import torch.nn as nn
import torch.nn.functional as F


class global_enhancev2(nn.Module):
    def __init__(self,input_dim=256) -> None:
        super().__init__()
        alpha=torch.tensor([1],dtype=torch.float32)
        self.alpha=torch.nn.parameter.Parameter(data=alpha,requires_grad=True)
        beta=torch.tensor([0],dtype=torch.float32)
        self.beta=torch.nn.parameter.Parameter(data=beta,requires_grad=True)
        self.linear=torch.nn.Conv1d(input_dim,input_dim,1,bias=False)

    def forward(self,feat):
        '''
        feat:B,N,D
        '''
        global_feat=torch.nn.functional.adaptive_avg_pool1d(feat.transpose(1,2),1) #B,D,1
        global_feat=self.linear(global_feat)
        atten=torch.bmm(feat,global_feat) #B,N,1
        mean=atten.squeeze(-1).mean(-1) #B
        std=torch.std(atten.squeeze(-1), dim=-1, unbiased=False) 
        atten=self.alpha*(atten-mean.view(-1,1,1))/(std.view(-1,1,1)+1e-5)+self.beta
        global_feat=torch.bmm(atten,global_feat.transpose(1,2)) #B,N,D
        feat=feat+global_feat #B,N,D
        return feat