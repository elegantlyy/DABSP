from collections import OrderedDict
from os.path import join
import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from models.model_utils import *

"""
# https://github.com/mahmoodlab/CLAM/blob/master/models/model_clam.py
Implement Attention MIL for the unimodal (WSI only) and multimodal setting (pathways + WSI). 
The combining of modalities can be done using bilinear fusion or concatenation. 
"""
class Attn_Net(nn.Module):

    def __init__(self, L = 1024, D = 256, dropout = False, n_classes = 1):
        super(Attn_Net, self).__init__()
        self.module = [
            nn.Linear(L, D),
            nn.Tanh()]

        if dropout:
            self.module.append(nn.Dropout(0.25))

        self.module.append(nn.Linear(D, n_classes))
        
        self.module = nn.Sequential(*self.module)
    
    def forward(self, x):
        return self.module(x), x # N x n_classes

"""
Attention Network with Sigmoid Gating (3 fc layers)
args:
    L: input feature dimension
    D: hidden layer dimension
    dropout: whether to use dropout (p = 0.25)
    n_classes: number of classes 
"""
class Attn_Net_Gated(nn.Module):
    def __init__(self, L = 1024, D = 256, dropout = False, n_classes = 1):
        super(Attn_Net_Gated, self).__init__()
        self.attention_a = [
            nn.Linear(L, D),
            nn.Tanh()]
        
        self.attention_b = [nn.Linear(L, D),
                            nn.Sigmoid()]
        if dropout:
            self.attention_a.append(nn.Dropout(0.25))
            self.attention_b.append(nn.Dropout(0.25))

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)
        
        self.attention_c = nn.Linear(D, n_classes)

    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)  # N x n_classes
        return A, x

class CLAM_SB(nn.Module):
    def __init__(self, gate = True, size_arg = "small", dropout = 0., k_sample=8, n_classes=2,
        instance_loss_fn=nn.CrossEntropyLoss(), subtyping=False, embed_dim=1024):
        super().__init__()
        self.size_dict = {"small": [embed_dim, 512, 256], "big": [embed_dim, 512, 384]}
        size = self.size_dict[size_arg]
        fc = [nn.Linear(size[0], size[1]), nn.ReLU(), nn.Dropout(dropout)]
        if gate:
            attention_net = Attn_Net_Gated(L = size[1], D = size[2], dropout = dropout, n_classes = 1)
        else:
            attention_net = Attn_Net(L = size[1], D = size[2], dropout = dropout, n_classes = 1)
        fc.append(attention_net)
        self.attention_net = nn.Sequential(*fc)
        self.classifiers = nn.Linear(size[1], n_classes)
        instance_classifiers = [nn.Linear(size[1], 2) for i in range(n_classes)]
        self.instance_classifiers = nn.ModuleList(instance_classifiers)
        self.k_sample = k_sample
        self.instance_loss_fn = instance_loss_fn
        self.n_classes = n_classes
        self.subtyping = subtyping
    
    @staticmethod
    def create_positive_targets(length, device):
        return torch.full((length, ), 1, device=device).long()
    
    @staticmethod
    def create_negative_targets(length, device):
        return torch.full((length, ), 0, device=device).long()
    
    #instance-level evaluation for in-the-class attention branch
    def inst_eval(self, A, h, classifier): 
        device=h.device
        if len(A.shape) == 1:
            A = A.view(1, -1)
        top_p_ids = torch.topk(A, self.k_sample)[1][-1]
        top_p = torch.index_select(h, dim=0, index=top_p_ids)
        top_n_ids = torch.topk(-A, self.k_sample, dim=1)[1][-1]
        top_n = torch.index_select(h, dim=0, index=top_n_ids)
        p_targets = self.create_positive_targets(self.k_sample, device)
        n_targets = self.create_negative_targets(self.k_sample, device)

        all_targets = torch.cat([p_targets, n_targets], dim=0)
        all_instances = torch.cat([top_p, top_n], dim=0)
        logits = classifier(all_instances)
        all_preds = torch.topk(logits, 1, dim = 1)[1].squeeze(1)
        instance_loss = self.instance_loss_fn(logits, all_targets)
        return instance_loss, all_preds, all_targets
    
    #instance-level evaluation for out-of-the-class attention branch
    def inst_eval_out(self, A, h, classifier):
        device=h.device
        if len(A.shape) == 1:
            A = A.view(1, -1)
        top_p_ids = torch.topk(A, self.k_sample)[1][-1]
        top_p = torch.index_select(h, dim=0, index=top_p_ids)
        p_targets = self.create_negative_targets(self.k_sample, device)
        logits = classifier(top_p)
        p_preds = torch.topk(logits, 1, dim = 1)[1].squeeze(1)
        instance_loss = self.instance_loss_fn(logits, p_targets)
        return instance_loss, p_preds, p_targets

    def forward(self, h, label=None, instance_eval=False, return_features=False, attention_only=False):
        A, h = self.attention_net(h)  # NxK        
        A = torch.transpose(A, 1, 0)  # KxN
        if attention_only:
            return A
        A_raw = A
        A = F.softmax(A, dim=1)  # softmax over N

        if instance_eval:
            total_inst_loss = 0.0
            all_preds = []
            all_targets = []
            inst_labels = F.one_hot(label, num_classes=self.n_classes).squeeze() #binarize label
            for i in range(len(self.instance_classifiers)):
                inst_label = inst_labels[i].item()
                classifier = self.instance_classifiers[i]
                if inst_label == 1: #in-the-class:
                    instance_loss, preds, targets = self.inst_eval(A, h, classifier)
                    all_preds.extend(preds.cpu().numpy())
                    all_targets.extend(targets.cpu().numpy())
                else: #out-of-the-class
                    if self.subtyping:
                        instance_loss, preds, targets = self.inst_eval_out(A, h, classifier)
                        all_preds.extend(preds.cpu().numpy())
                        all_targets.extend(targets.cpu().numpy())
                    else:
                        continue
                total_inst_loss += instance_loss

            if self.subtyping:
                total_inst_loss /= len(self.instance_classifiers)
                
        M = torch.mm(A, h) 
        logits = self.classifiers(M)
        Y_hat = torch.topk(logits, 1, dim = 1)[1]
        Y_prob = F.softmax(logits, dim = 1)
        if instance_eval:
            results_dict = {'instance_loss': total_inst_loss, 'inst_labels': np.array(all_targets), 
            'inst_preds': np.array(all_preds)}
        else:
            results_dict = {}
        if return_features:
            results_dict.update({'features': M})
        return logits, Y_prob, Y_hat, A_raw, results_dict

class CLAM_MB(CLAM_SB):
    def __init__(self, gate = True, size_arg = "small", dropout = 0., k_sample=8, n_classes=2,
        instance_loss_fn=nn.CrossEntropyLoss(), subtyping=False, embed_dim=1024):
        nn.Module.__init__(self)
        self.size_dict = {"small": [embed_dim, 512, 256], "big": [embed_dim, 512, 384]}
        size = self.size_dict[size_arg]
        fc = [nn.Linear(size[0], size[1]), nn.ReLU(), nn.Dropout(dropout)]
        if gate:
            attention_net = Attn_Net_Gated(L = size[1], D = size[2], dropout = dropout, n_classes = n_classes)
        else:
            attention_net = Attn_Net(L = size[1], D = size[2], dropout = dropout, n_classes = n_classes)
        fc.append(attention_net)
        self.attention_net = nn.Sequential(*fc)
        bag_classifiers = [nn.Linear(size[1], 1) for i in range(n_classes)] #use an indepdent linear layer to predict each class
        self.classifiers = nn.ModuleList(bag_classifiers)
        instance_classifiers = [nn.Linear(size[1], 2) for i in range(n_classes)]
        self.instance_classifiers = nn.ModuleList(instance_classifiers)
        self.k_sample = k_sample
        self.instance_loss_fn = instance_loss_fn
        self.n_classes = n_classes
        self.subtyping = subtyping

    def forward(self, h, label=None, instance_eval=False, return_features=False, attention_only=False):
        A, h = self.attention_net(h)  # NxK        
        A = torch.transpose(A, 1, 0)  # KxN
        if attention_only:
            return A
        A_raw = A
        A = F.softmax(A, dim=1)  # softmax over N

        if instance_eval:
            total_inst_loss = 0.0
            all_preds = []
            all_targets = []
            inst_labels = F.one_hot(label, num_classes=self.n_classes).squeeze() #binarize label
            for i in range(len(self.instance_classifiers)):
                inst_label = inst_labels[i].item()
                classifier = self.instance_classifiers[i]
                if inst_label == 1: #in-the-class:
                    instance_loss, preds, targets = self.inst_eval(A[i], h, classifier)
                    all_preds.extend(preds.cpu().numpy())
                    all_targets.extend(targets.cpu().numpy())
                else: #out-of-the-class
                    if self.subtyping:
                        instance_loss, preds, targets = self.inst_eval_out(A[i], h, classifier)
                        all_preds.extend(preds.cpu().numpy())
                        all_targets.extend(targets.cpu().numpy())
                    else:
                        continue
                total_inst_loss += instance_loss

            if self.subtyping:
                total_inst_loss /= len(self.instance_classifiers)

        M = torch.mm(A, h) 

        logits = torch.empty(1, self.n_classes).float().to(M.device)
        for c in range(self.n_classes):
            logits[0, c] = self.classifiers[c](M[c])

        Y_hat = torch.topk(logits, 1, dim = 1)[1]
        Y_prob = F.softmax(logits, dim = 1)
        if instance_eval:
            results_dict = {'instance_loss': total_inst_loss, 'inst_labels': np.array(all_targets), 
            'inst_preds': np.array(all_preds)}
        else:
            results_dict = {}
        if return_features:
            results_dict.update({'features': M})
        return logits, Y_prob, Y_hat, A_raw, results_dict


class CLAM(nn.Module):
    def __init__(self, fusion=None, size_arg="small", dropout=0.25, n_classes=4, gate=True,
                 embed_dim=1024, df_comp=None, omic_input_dim=None,
                 dim_per_path_1=16, dim_per_path_2=64, device="cpu"):
        super().__init__()
        self.fusion = fusion
        self.device = device
        self.use_omics = fusion is not None
        self.df_comp = df_comp
        self.dim_per_path_1 = dim_per_path_1
        self.dim_per_path_2 = dim_per_path_2
        self.omic_input_dim = omic_input_dim

        # --- 1. Omics pathway encoder ---
        if self.use_omics:
            self.num_pathways = self.df_comp.shape[1]
            M_raw = torch.Tensor(self.df_comp.values)
            self.mask_1 = torch.repeat_interleave(M_raw, self.dim_per_path_1, dim=1)
            self.fc_1_weight = nn.Parameter(torch.empty(self.omic_input_dim, self.dim_per_path_1 * self.num_pathways))
            self.fc_1_bias = nn.Parameter(torch.rand(self.dim_per_path_1 * self.num_pathways))
            nn.init.xavier_normal_(self.fc_1_weight)

            self.fc_2_weight = nn.Parameter(torch.empty(self.dim_per_path_1 * self.num_pathways, self.dim_per_path_2 * self.num_pathways))
            self.fc_2_bias = nn.Parameter(torch.rand(self.dim_per_path_2 * self.num_pathways))
            nn.init.xavier_normal_(self.fc_2_weight)

            # Create mask_2
            mask_2_np = np.zeros([self.dim_per_path_1 * self.num_pathways, self.dim_per_path_2 * self.num_pathways])
            for i in range(self.num_pathways):
                row = i * self.dim_per_path_1
                col = i * self.dim_per_path_2
                mask_2_np[row:row + self.dim_per_path_1, col:col + self.dim_per_path_2] = 1
            self.mask_2 = torch.Tensor(mask_2_np)

            self.upscale = nn.Sequential(
                nn.Linear(self.dim_per_path_2 * self.num_pathways, 256),
                nn.ReLU(),
                nn.Dropout(dropout)
            )

            
            self.project_h_wsi = nn.Identity()  

            if fusion == "concat":
                self.mm = nn.Sequential(
                    nn.Linear(256*2, 256),
                    nn.ReLU(),
                    nn.Linear(256, 256),
                    nn.ReLU()
                )
            elif fusion == "bilinear":
                self.mm = BilinearFusion(dim1=256, dim2=256, scale_dim1=8, scale_dim2=8, mmhid=256)
            else:
                raise ValueError("Unsupported fusion method.")

        # --- 2. CLAM backbone ---
        self.clam_model = CLAM_SB(
            gate=gate,
            size_arg=size_arg,
            dropout=dropout,
            n_classes=n_classes,
            subtyping=False,
            embed_dim=embed_dim
        )

        self.classifier = nn.Linear(256, n_classes)
        self.classifier = self.classifier.to(self.device)

    def maybe_init_projector(self, h_wsi):
        if isinstance(self.project_h_wsi, nn.Identity):
            in_dim = h_wsi.shape[1]
            self.project_h_wsi = nn.Linear(in_dim, 256).to(self.device)


    def relocate(self):
        """
        Multi-GPU relocation
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.device_count() > 1:
            self.clam_model = nn.DataParallel(self.clam_model)
            if self.use_omics:
                self.mm = nn.DataParallel(self.mm)
                self.upscale = nn.DataParallel(self.upscale)
        self.to(device)

    def forward(self, **kwargs):
        x_wsi = kwargs["data_WSI"].squeeze().to(self.device)  # [N, 1024]
        return_features = self.use_omics

        # --- 1. Get WSI embedding from CLAM ---
        logits, _, _, _, results = self.clam_model(x_wsi, return_features=return_features)

        if self.use_omics:
            x_omics = kwargs["data_omics"].squeeze().to(self.device)
            h_wsi = results["features"] 
            
            self.maybe_init_projector(h_wsi)
            h_wsi = self.project_h_wsi(h_wsi) # [1, 256]

            # Omics encoder
            out = torch.matmul(x_omics, self.fc_1_weight * self.mask_1.to(self.device)) + self.fc_1_bias
            out = F.relu(out)
            out = torch.matmul(out, self.fc_2_weight * self.mask_2.to(self.device)) + self.fc_2_bias
            h_omics = self.upscale(out).unsqueeze(0)  # [1, 256]

            # Fusion
            if self.fusion == "concat":
                h_fused = self.mm(torch.cat([h_wsi, h_omics], dim=1))
            elif self.fusion == "bilinear":
                h_fused = self.mm(h_wsi, h_omics)

            logits = self.classifier(h_fused)

        return logits  
