from ast import Lambda
import numpy as np
import pdb
import os
from custom_optims.radam import RAdam
from models.model_ABMIL import ABMIL
from models.model_CLAM import CLAM 
from models.model_DeepMISL import DeepMISL
from models.model_MLPOmics import MLPOmics
from models.model_MLPWSI import MLPWSI #
from models.model_SNNOmics import SNNOmics
from models.model_MaskedOmics import MaskedOmics

from models.model_MCATPathways import MCATPathways
from models.model_MOTCat import MOTCAT_Surv # MCATPathwaysMotCat

from models.model_SurvPath import SurvPath, SurvPath_ctranspath
from models.model_DualAttnLora import SNNChebyDualAttnLora # with lora and residual
from models.model_DualAttnLora_wo_residual import SNNChebyDualAttnLoraWoRes # w lora, w/o residual
from models.model_DualAttnLora_wo_snn import SNNChebyDualAttnLoraWoSNN # w lora, w/o snn
from models.model_DualAttnLora_wo_chebykan import SNNChebyDualAttnLoraWoChebyKan # w lora, w/o chebykan
from models.model_DualAttnLora_wo_attn_use_cat_or_kp import SNNChebyDualAttnLoraWoAttn # w lora, w/o attn, use cat or kp
from models.model_DualAttnLora_one_modality import SNNChebyDualAttnLoraOneModality #w  lora, wo_omics or wo_wsi
from models.model_DualAttnLora_uniattn import SNNChebyUniAttnLora # w lora , uni-directional attention
from models.model_DualAttnLora_wo_lora import SNNChebyDualAttn #  wo_lora, use nn.linear for q and kv
from models.model_TMIL import TMIL
from models.model_utils import * # BilinearFusion, SNN_Block, Reg_Block, Attn_Net_Gated, init_max_weights
# import sksurv
from sksurv.metrics import concordance_index_censored, concordance_index_ipcw, brier_score, integrated_brier_score, cumulative_dynamic_auc
from sksurv.util import Surv

from transformers import (
    get_constant_schedule_with_warmup, 
    get_linear_schedule_with_warmup, 
    get_cosine_schedule_with_warmup
)

import captum
import pickle
from captum.attr import LayerIntegratedGradients, TokenReferenceBase, visualization
from captum.attr import IntegratedGradients
import matplotlib.pyplot as plt

#----> pytorch imports
import torch
from torch.nn.utils.rnn import pad_sequence

from utils.general_utils import _get_split_loader, _print_network, _save_splits
from utils.loss_func import NLLSurvLoss, CombinedLoss, MAELossFunction, NllAddOrthogonalSurvLoss

import torch.optim as optim



def _get_splits(datasets, cur, args):
    r"""
    Summarize the train and val splits and return them individually
    
    Args:
        - datasets : tuple
        - cur : Int 
        - args: argspace.Namespace
    
    Return:
        - train_split : SurvivalDataset
        - val_split : SurvivalDataset
    
    """

    print('\nTraining Fold {}!'.format(cur))
    print('\nInit train/val splits...', end=' ')
    train_split, val_split = datasets
    _save_splits(datasets, ['train', 'val'], os.path.join(args.results_dir, 'splits_{}.csv'.format(cur)))
    print('Done!')
    print("Training on {} samples".format(len(train_split)))
    print("Validating on {} samples".format(len(val_split)))

    return train_split,val_split


# def _init_loss_function(args, model):
def _init_loss_function(args):
    r"""
    Init the survival loss function
    
    Args:
        - args : argspace.Namespace 
    
    Returns:
        - loss_fn : NLLSurvLoss or NLLRankSurvLoss or CombinedLoss
    """

    print('\nInit loss function...', end=' ')
    if args.bag_loss == 'nll_surv':
        loss_fn = NLLSurvLoss(alpha=args.alpha_surv)
    elif args.bag_loss == 'nll_surv_mse':
        nll_loss_instance = NLLSurvLoss(alpha=args.alpha_surv)
        mae_loss_instance = MAELossFunction()
        loss_fn = CombinedLoss(nll_loss_instance, mae_loss_instance, nll_weight=args.nll_weight, mae_weight=args.mae_weight)
    elif args.bag_loss == 'nll_ortho_surv':    
        intermediate_results = model.get_intermediate_results()
        paths_postSA_embed = intermediate_results['paths_postSA_embed']
        wsi_postSA_embed = intermediate_results['wsi_postSA_embed']
        loss_fn = NllAddOrthogonalSurvLoss(paths_postSA_embed=paths_postSA_embed, wsi_postSA_embed=wsi_postSA_embed)
        
    else:
        raise NotImplementedError
    print('Done!')
    return loss_fn

def _init_optim(args, model):
    r"""
    Init the optimizer 
    
    Args: 
        - args : argspace.Namespace 
        - model : torch model 
    
    Returns:
        - optimizer : torch optim 
    """
    print('\nInit optimizer ...', end=' ')

    if args.opt == "adam":
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    elif args.opt == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.reg)
    elif args.opt == "adamW":
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.reg)
    elif args.opt == "radam":
        optimizer = RAdam(model.parameters(), lr=args.lr, weight_decay=args.reg)
    elif args.opt == "lamb":
        optimizer = Lambda(model.parameters(), lr=args.lr, weight_decay=args.reg)
    else:
        raise NotImplementedError

    return optimizer

def _init_model(args):
    
    print('\nInit Model...', end=' ')
    if args.type_of_path == "xena":
        omics_input_dim = 1577
    elif args.type_of_path == "hallmarks":
        omics_input_dim = 4241
    elif args.type_of_path == "combine":
        omics_input_dim = 4999
    elif args.type_of_path == "multi":
        if args.study == "tcga_brca":
            omics_input_dim = 9947
        else:
            omics_input_dim = 14933
    else:
        omics_input_dim = 0
    
    # omics baselines
    if args.modality == "mlp_per_path":

        model_dict = {
            "device" : args.device, "df_comp" : args.composition_df, "input_dim" : omics_input_dim,
            "dim_per_path_1" : args.encoding_layer_1_dim, "dim_per_path_2" : args.encoding_layer_2_dim,
            "dropout" : args.encoder_dropout, "num_classes" : args.n_classes,
        }
        model = MaskedOmics(**model_dict)

    elif args.modality == "omics":

        model_dict = {
             "input_dim" : omics_input_dim, "projection_dim": 64, "dropout": args.encoder_dropout
        }
        model = MLPOmics(**model_dict)

    elif args.modality == "snn":

        model_dict = {
             "omic_input_dim" : omics_input_dim, 
        }
        model = SNNOmics(**model_dict)

    elif args.modality in ["abmil_wsi", "abmil_wsi_pathways"]:

        model_dict = {
            "device" : args.device, "df_comp" : args.composition_df, "omic_input_dim" : omics_input_dim,
            "dim_per_path_1" : args.encoding_layer_1_dim, "dim_per_path_2" : args.encoding_layer_2_dim,
            "fusion":args.fusion
        }

        model = ABMIL(**model_dict)
    
    elif args.modality in ["clam_wsi", "clam_wsi_pathways"]:

        model_dict = {
            "device" : args.device, "df_comp" : args.composition_df, "omic_input_dim" : omics_input_dim,
            "dim_per_path_1" : args.encoding_layer_1_dim, "dim_per_path_2" : args.encoding_layer_2_dim,
            "fusion":args.fusion
        }

        model = CLAM(**model_dict)


    # unimodal and multimodal baselines
    elif args.modality in ["deepmisl_wsi", "deepmisl_wsi_pathways"]:

        model_dict = {
            "device" : args.device, "df_comp" : args.composition_df, "omic_input_dim" : omics_input_dim,
            "dim_per_path_1" : args.encoding_layer_1_dim, "dim_per_path_2" : args.encoding_layer_2_dim,
            "fusion":args.fusion
        }

        model = DeepMISL(**model_dict)

    elif args.modality == "mlp_wsi":
        
        model_dict = {
            "wsi_embedding_dim":args.encoding_dim, "input_dim_omics":omics_input_dim, "dropout":args.encoder_dropout,
            "device": args.device

        }
        model = MLPWSI(**model_dict)

    elif args.modality in ["transmil_wsi", "transmil_wsi_pathways"]:

        model_dict = {
            "device" : args.device, "df_comp" : args.composition_df, "omic_input_dim" : omics_input_dim,
            "dim_per_path_1" : args.encoding_layer_1_dim, "dim_per_path_2" : args.encoding_layer_2_dim,
            "fusion":args.fusion
        }

        model = TMIL(**model_dict)

    elif args.modality == "coattn":

        model_dict = {'fusion': args.fusion, 'omic_sizes': args.omic_sizes, 'n_classes': args.n_classes}
        model = MCATPathways(**model_dict)

    elif args.modality == "coattn_motcat":

        model_dict = {
            'fusion': args.fusion, 'omic_sizes': args.omic_sizes, 'n_classes': args.n_classes,
            "ot_reg":0.1, "ot_tau":0.5, "ot_impl":"pot-uot-l2"
        }
        # model = MCATPathwaysMotCat(**model_dict)#MOTCAT_Surv
        model = MOTCAT_Surv(**model_dict)#MOTCAT_Surv

    # survpath 
    elif args.modality == "survpath":

        model_dict = {'wsi_embedding_dim': args.encoding_dim, 'omic_sizes': args.omic_sizes, 'num_classes': args.n_classes,
                      'wsi_projection_dim': args.wsi_projection_dim} # 768, 4999， 4

        if args.use_nystrom:
            model = SurvPath_with_nystrom(**model_dict)
        else:
            model = SurvPath(**model_dict)

# --------------------------------------------------------------------------------------------------------------------- # 
    elif args.modality in ["snn_chebykan_snn_dual_attn_lora"]:

        model_dict = {'wsi_embedding_dim': args.encoding_dim, 'omic_sizes': args.omic_sizes, 'num_classes': args.n_classes,
                      'per_pathway_gene_number_over_n': args.per_pathway_over_n,
                      'cheby_hidden': args.cheby_hidden, 'cheby_degree': args.cheby_degree,
                      'wsi_projection_dim': args.wsi_projection_dim,
                      'lora_r': args.lora_r, 'lora_alpha': args.lora_alpha
                      } 
        
        model = SNNChebyDualAttnLora(**model_dict)   
# ---------------------------------------------------- w/o  ----------------------------------------------------------------- # 
    elif args.modality in ["snn_chebykan_snn_dual_attn_lora_wo_residual"]:
        # wo_res
        model_dict = {'wsi_embedding_dim': args.encoding_dim, 'omic_sizes': args.omic_sizes, 'num_classes': args.n_classes,
                      'per_pathway_gene_number_over_n': args.per_pathway_over_n,
                      'cheby_hidden': args.cheby_hidden, 'cheby_degree': args.cheby_degree,
                      'wsi_projection_dim': args.wsi_projection_dim,
                      'lora_r': args.lora_r, 'lora_alpha': args.lora_alpha
                      } 
        
        model = SNNChebyDualAttnLoraWoRes(**model_dict)   

    elif args.modality in ["snn_chebykan_snn_dual_attn_lora_wo_snn"]:
        # wo_snn
        model_dict = {'wsi_embedding_dim': args.encoding_dim, 'omic_sizes': args.omic_sizes, 'num_classes': args.n_classes,
                      'per_pathway_gene_number_over_n': args.per_pathway_over_n,
                      'cheby_hidden': args.cheby_hidden, 'cheby_degree': args.cheby_degree,
                      'wsi_projection_dim': args.wsi_projection_dim,
                      'lora_r': args.lora_r, 'lora_alpha': args.lora_alpha
                      } 
        
        model = SNNChebyDualAttnLoraWoSNN(**model_dict)   


    elif args.modality in ["snn_chebykan_snn_dual_attn_lora_wo_chebykan"]:
        # wo_chebykan
        model_dict = {'wsi_embedding_dim': args.encoding_dim, 'omic_sizes': args.omic_sizes, 'num_classes': args.n_classes,
                      'per_pathway_gene_number_over_n': args.per_pathway_over_n,
                      'cheby_hidden': args.cheby_hidden, 'cheby_degree': args.cheby_degree,
                      'wsi_projection_dim': args.wsi_projection_dim,
                      'lora_r': args.lora_r, 'lora_alpha': args.lora_alpha
                      } 
        
        model = SNNChebyDualAttnLoraWoChebyKan(**model_dict)   

    elif args.modality in ["snn_chebykan_snn_dual_attn_lora_wo_attn"]:
        # wo_chebykan
        model_dict = {'wsi_embedding_dim': args.encoding_dim, 'omic_sizes': args.omic_sizes, 'num_classes': args.n_classes,
                      'per_pathway_gene_number_over_n': args.per_pathway_over_n,
                      'cheby_hidden': args.cheby_hidden, 'cheby_degree': args.cheby_degree,
                      'wsi_projection_dim': args.wsi_projection_dim,
                      'our_fusion': args.our_fusion, # concat or kp
                      } 
        
        model = SNNChebyDualAttnLoraWoAttn(**model_dict)   


    elif args.modality in ["snn_chebykan_snn_dual_attn_lora_wo_omics", "snn_chebykan_snn_dual_attn_lora_wo_wsi"]:
        # one modality
        model_dict = {'wsi_embedding_dim': args.encoding_dim, 'omic_sizes': args.omic_sizes, 'num_classes': args.n_classes,
                      'per_pathway_gene_number_over_n': args.per_pathway_over_n,
                      'cheby_hidden': args.cheby_hidden, 'cheby_degree': args.cheby_degree,
                      'wsi_projection_dim': args.wsi_projection_dim,
                      'wo_omics': args.wo_omics, 
                      'wo_wsi': args.wo_wsi, 
                      } 
        
        model = SNNChebyDualAttnLoraOneModality(**model_dict)   

    elif args.modality in ["snn_chebykan_snn_dual_attn_lora_wo_biattn"]:
        # wo_bi_directional_attention, use_uni_directional
        model_dict = {'wsi_embedding_dim': args.encoding_dim, 'omic_sizes': args.omic_sizes, 'num_classes': args.n_classes,
                      'per_pathway_gene_number_over_n': args.per_pathway_over_n,
                      'cheby_hidden': args.cheby_hidden, 'cheby_degree': args.cheby_degree,
                      'wsi_projection_dim': args.wsi_projection_dim,
                      } 
        
        model = SNNChebyUniAttnLora(**model_dict)   


    elif args.modality in ["snn_chebykan_snn_dual_attn_lora_wo_lora"]:
        # wo_lora, cross-attn use nn.linear for q,kv
        model_dict = {'wsi_embedding_dim': args.encoding_dim, 'omic_sizes': args.omic_sizes, 'num_classes': args.n_classes,
                      'per_pathway_gene_number_over_n': args.per_pathway_over_n,
                      'cheby_hidden': args.cheby_hidden, 'cheby_degree': args.cheby_degree,
                      'wsi_projection_dim': args.wsi_projection_dim,
                      } 
        
        model = SNNChebyDualAttn(**model_dict)   


    else:
        raise NotImplementedError

    if torch.cuda.is_available():
        model = model.to(torch.device('cuda'))

    print('Done!')
    _print_network(args.results_dir, model)

    return model



def _init_loaders(args, train_split, val_split):
    r"""
    Init dataloaders for the train and val datasets 

    Args:
        - args : argspace.Namespace 
        - train_split : SurvivalDataset 
        - val_split : SurvivalDataset 
    
    Returns:
        - train_loader : Pytorch Dataloader 
        - val_loader : Pytorch Dataloader

    """

    print('\nInit Loaders...', end=' ')
    if train_split:
        train_loader = _get_split_loader(args, train_split, training=True, testing=False, weighted=args.weighted_sample, batch_size=args.batch_size)
    else:
        train_loader = None

    if val_split:
        val_loader = _get_split_loader(args, val_split,  testing=False, batch_size=1)
    else:
        val_loader = None
    print('Done!')

    return train_loader,val_loader

def _extract_survival_metadata(train_loader, val_loader):
    r"""
    Extract censorship and survival times from the train and val loader and combine to get numbers for the fold
    We need to do this for train and val combined because when evaulating survival metrics, the function needs to know the 
    distirbution of censorhsip and survival times for the trainig data
    
    Args:
        - train_loader : Pytorch Dataloader
        - val_loader : Pytorch Dataloader
    
    Returns:
        - all_survival : np.array
    
    """

    all_censorships = np.concatenate(
        [train_loader.dataset.metadata[train_loader.dataset.censorship_var].to_numpy(),
        val_loader.dataset.metadata[val_loader.dataset.censorship_var].to_numpy()],
        axis=0)

    all_event_times = np.concatenate(
        [train_loader.dataset.metadata[train_loader.dataset.label_col].to_numpy(),
        val_loader.dataset.metadata[val_loader.dataset.label_col].to_numpy()],
        axis=0)

    all_survival = Surv.from_arrays(event=(1-all_censorships).astype(bool), time=all_event_times)
    return all_survival

def _unpack_data(modality, device, data):
    r"""
    Depending on the model type, unpack the data and put it on the correct device
    
    Args:
        - modality : String 
        - device : torch.device 
        - data : tuple 
    
    Returns:
        - data_WSI : torch.Tensor
        - mask : torch.Tensor
        - y_disc : torch.Tensor
        - event_time : torch.Tensor
        - censor : torch.Tensor
        - data_omics : torch.Tensor
        - clinical_data_list : list
        - mask : torch.Tensor
    
    """
    
    if modality in ["mlp_per_path", "omics", "snn"]:
        data_WSI = data[0]
        mask = None
        data_omics = data[1].to(device)
        y_disc, event_time, censor, clinical_data_list = data[2], data[3], data[4], data[5]
    
    elif modality in ["mlp_per_path_wsi", "abmil_wsi", "abmil_wsi_pathways", "clam_wsi", "clam_wsi_pathways", "deepmisl_wsi", "deepmisl_wsi_pathways", "mlp_wsi", "transmil_wsi", "transmil_wsi_pathways"]:
        data_WSI = data[0].to(device)
        data_omics = data[1].to(device)
        
        if data[6][0,0] == 1:
            mask = None
        else:
            mask = data[6].to(device)

        y_disc, event_time, censor, clinical_data_list = data[2], data[3], data[4], data[5]

    elif modality in ["coattn", "coattn_motcat"]:
        
        data_WSI = data[0].to(device)
        data_omic1 = data[1].type(torch.FloatTensor).to(device)
        data_omic2 = data[2].type(torch.FloatTensor).to(device)
        data_omic3 = data[3].type(torch.FloatTensor).to(device)
        data_omic4 = data[4].type(torch.FloatTensor).to(device)
        data_omic5 = data[5].type(torch.FloatTensor).to(device)
        data_omic6 = data[6].type(torch.FloatTensor).to(device)
        data_omics = [data_omic1, data_omic2, data_omic3, data_omic4, data_omic5, data_omic6]

        y_disc, event_time, censor, clinical_data_list, mask = data[7], data[8], data[9], data[10], data[11]
        mask = mask.to(device)

    elif modality in ["snn_chebykan_snn_dual_attn_lora",
                      "snn_chebykan_snn_dual_attn_lora_wo_residual",
                      "snn_chebykan_snn_dual_attn_lora_wo_snn",
                      "snn_chebykan_snn_dual_attn_lora_wo_chebykan",
                      "snn_chebykan_snn_dual_attn_lora_wo_attn",
                      "snn_chebykan_snn_dual_attn_lora_wo_omics",
                      "snn_chebykan_snn_dual_attn_lora_wo_wsi",
                      "snn_chebykan_snn_dual_attn_lora_wo_biattn",
                      "snn_chebykan_snn_dual_attn_lora_wo_lora"]:

        data_WSI = data[0].to(device)

        data_omics = []
        for item in data[1][0]:
            data_omics.append(item.to(device))
        
        if data[6][0,0] == 1:
            mask = None
        else:
            mask = data[6].to(device)

        y_disc, event_time, censor, clinical_data_list = data[2], data[3], data[4], data[5]
        
    else:
        raise ValueError('Unsupported modality:', modality)
    
    y_disc, event_time, censor = y_disc.to(device), event_time.to(device), censor.to(device)

    return data_WSI, mask, y_disc, event_time, censor, data_omics, clinical_data_list, mask

def _process_data_and_forward(model, modality, device, data):
    r"""
    Depeding on the modality, process the input data and do a forward pass on the model 
    
    Args:
        - model : Pytorch model
        - modality : String
        - device : torch.device
        - data : tuple
    
    Returns:
        - out : torch.Tensor
        - y_disc : torch.Tensor
        - event_time : torch.Tensor
        - censor : torch.Tensor
        - clinical_data_list : List
    
    """
    data_WSI, mask, y_disc, event_time, censor, data_omics, clinical_data_list, mask = _unpack_data(modality, device, data)

    if modality in ["coattn", "coattn_motcat"]:  
        
        out = model(
            x_path=data_WSI, 
            x_omic1=data_omics[0], 
            x_omic2=data_omics[1], 
            x_omic3=data_omics[2], 
            x_omic4=data_omics[3], 
            x_omic5=data_omics[4], 
            x_omic6=data_omics[5]
            )  

    elif modality in ("snn_chebykan_snn_dual_attn_lora",
                      "snn_chebykan_snn_dual_attn_lora_wo_residual",
                      "snn_chebykan_snn_dual_attn_lora_wo_snn",
                      "snn_chebykan_snn_dual_attn_lora_wo_chebykan",
                      "snn_chebykan_snn_dual_attn_lora_wo_attn",
                      "snn_chebykan_snn_dual_attn_lora_wo_omics",
                      "snn_chebykan_snn_dual_attn_lora_wo_wsi",
                      "snn_chebykan_snn_dual_attn_lora_wo_biattn",
                      "snn_chebykan_snn_dual_attn_lora_wo_lora",
                      ):

        input_args = {"x_path": data_WSI.to(device)}
        for i in range(len(data_omics)): # 0-330
            input_args['x_omic%s' % str(i+1)] = data_omics[i].type(torch.FloatTensor).to(device) # 1-331
        input_args["return_attn"] = False
        out = model(**input_args)

    else:
        out = model(
            data_omics = data_omics, 
            data_WSI = data_WSI, 
            mask = mask
            )
        
    if len(out.shape) == 1:
            out = out.unsqueeze(0)
    return out, y_disc, event_time, censor, clinical_data_list


def _calculate_risk(h):
    r"""
    Take the logits of the model and calculate the risk for the patient 
    
    Args: 
        - h : torch.Tensor , h=model(**input_args)
    
    Returns:
        - risk : torch.Tensor 
    
    """
    hazards = torch.sigmoid(h) # h_hat # tensor: (1,4) 
    survival = torch.cumprod(1 - hazards, dim=1) # 1-y_hat
    risk = -torch.sum(survival, dim=1).detach().cpu().numpy()
    return risk, survival.detach().cpu().numpy()

def _update_arrays(all_risk_scores, all_censorships, all_event_times, all_clinical_data, event_time, censor, risk, clinical_data_list):
    r"""
    Update the arrays with new values 
    
    Args:
        - all_risk_scores : List
        - all_censorships : List
        - all_event_times : List
        - all_clinical_data : List
        - event_time : torch.Tensor
        - censor : torch.Tensor
        - risk : torch.Tensor
        - clinical_data_list : List
    
    Returns:
        - all_risk_scores : List
        - all_censorships : List
        - all_event_times : List
        - all_clinical_data : List
    
    """
    all_risk_scores.append(risk)
    all_censorships.append(censor.detach().cpu().numpy())
    all_event_times.append(event_time.detach().cpu().numpy())
    all_clinical_data.append(clinical_data_list)
    return all_risk_scores, all_censorships, all_event_times, all_clinical_data


def MAE_with_censorship(all_hazards_which, y_disc_which, all_censorships):
    all_censorships = np.array(all_censorships)
    print("all_censorships", all_censorships.shape)
    valid_indices = np.where(all_censorships == 0)[0]  
    valid_hazards = all_hazards_which[valid_indices]
    valid_y_disc = y_disc_which[valid_indices]
    mae = np.mean(np.abs(valid_hazards - valid_y_disc))
    return mae


def orthogonal_loss(paths_postSA_embed, wsi_postSA_embed, lambda1=0.5, lambda2=0.5):
    embd1 = paths_postSA_embed
    embd2 = wsi_postSA_embed
    dot_product = torch.dot(embd1, embd2)
    loss = lambda1 * torch.abs(dot_product) + lambda2 * dot_product**2
    return loss


def _train_loop_survival(epoch, model, modality, loader, optimizer, scheduler, loss_fn):
    r"""
    Perform one epoch of training 

    Args:
        - epoch : Int
        - model : Pytorch model
        - modality : String 
        - loader : Pytorch dataloader
        - optimizer : torch.optim
        - loss_fn : custom loss function class 
    
    Returns:
        - c_index : Float
        - total_loss : Float 
    
    """
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(log_dir=f'./tensorboard') 

    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train()

    total_loss = 0.
    
    all_risk_scores = []
    all_censorships = []
    all_event_times = []
    all_clinical_data = []
    
    # mae_values = []

    # one epoch
    for batch_idx, data in enumerate(loader):
        
        optimizer.zero_grad()

        h, y_disc, event_time, censor, clinical_data_list = _process_data_and_forward(model, modality, device, data)
    
        loss = loss_fn(h=h, y=y_disc, t=event_time, c=censor)
        loss_value = loss.item()
        loss = loss / y_disc.shape[0]
        writer.add_scalar('Loss/train', loss, global_step=epoch)
        
        risk, _ = _calculate_risk(h) # risk, survival

        all_risk_scores, all_censorships, all_event_times, all_clinical_data = _update_arrays(all_risk_scores, all_censorships, all_event_times,all_clinical_data, event_time, censor, risk, clinical_data_list)
        

        total_loss += loss_value
        

        loss.backward()
        optimizer.step()
        scheduler.step()

        if (batch_idx % 20) == 0:
            print("batch: {}, loss: {:.3f}".format(batch_idx, loss.item()))
    
    total_loss /= len(loader.dataset)

    all_risk_scores = np.concatenate(all_risk_scores, axis=0)
    all_censorships = np.concatenate(all_censorships, axis=0)
    all_event_times = np.concatenate(all_event_times, axis=0)

    c_index = concordance_index_censored((1-all_censorships).astype(bool), all_event_times, all_risk_scores, tied_tol=1e-08)[0]
    print('Epoch: {}, train_loss: {:.4f}, train_c_index: {:.4f}'.format(epoch, total_loss, c_index))

    writer.close()
    attr_model = IntegratedGradients(model)

    return c_index, total_loss

def _calculate_metrics(loader, dataset_factory, survival_train, all_risk_scores, all_censorships, all_event_times, all_risk_by_bin_scores):
    r"""
    Calculate various survival metrics 
    
    Args:
        - loader : Pytorch dataloader
        - dataset_factory : SurvivalDatasetFactory
        - survival_train : np.array
        - all_risk_scores : np.array
        - all_censorships : np.array
        - all_event_times : np.array
        - all_risk_by_bin_scores : np.array
        
    Returns:
        - c_index : Float
        - c_index_ipcw : Float
        - BS : np.array
        - IBS : Float
        - iauc : Float
    
    """
    
    data = loader.dataset.metadata["survival_months_dss"]
    bins_original = dataset_factory.bins # 4
    which_times_to_eval_at = np.array([data.min() + 0.0001, bins_original[1], bins_original[2], data.max() - 0.0001])

    #---> delete the nans and corresponding elements from other arrays 
    original_risk_scores = all_risk_scores
    all_risk_scores = np.delete(all_risk_scores, np.argwhere(np.isnan(original_risk_scores)))
    all_censorships = np.delete(all_censorships, np.argwhere(np.isnan(original_risk_scores)))
    all_event_times = np.delete(all_event_times, np.argwhere(np.isnan(original_risk_scores)))
    #<---

    c_index = concordance_index_censored((1-all_censorships).astype(bool), all_event_times, all_risk_scores, tied_tol=1e-08)[0]
    c_index_ipcw, BS, IBS, iauc = 0., 0., 0., 0.

    # change the datatype of survival test to calculate metrics 
    try:
        survival_test = Surv.from_arrays(event=(1-all_censorships).astype(bool), time=all_event_times)
    except:
        print("Problem converting survival test datatype, so all metrics 0.")
        return c_index, c_index_ipcw, BS, IBS, iauc
   
    # cindex2 
    try:
        c_index_ipcw = concordance_index_ipcw(survival_train, survival_test, estimate=all_risk_scores)[0]
    except:
        print('An error occured while computing c-index ipcw')
        c_index_ipcw = 0.
    
    # brier score 
    try:
        _, BS = brier_score(survival_train, survival_test, estimate=all_risk_by_bin_scores, times=which_times_to_eval_at) # all_risk_by_bin_scores = survival， np.array([data.min() + 0.0001, bins_original[1], bins_original[2], data.max() - 0.0001])
    except:
        print('An error occured while computing BS')
        BS = 0.
    
    # IBS Brier
    try:
        IBS = integrated_brier_score(survival_train, survival_test, estimate=all_risk_by_bin_scores, times=which_times_to_eval_at)
    except:
        print('An error occured while computing IBS')
        IBS = 0.

    # iauc
    try:
        auc, iauc = cumulative_dynamic_auc(survival_train, survival_test, estimate=1-all_risk_by_bin_scores[:, 1:], times=which_times_to_eval_at[1:])
    except:
        print('An error occured while computing iauc')
        iauc = 0.
    
    return c_index, c_index_ipcw, BS, IBS, iauc


def _summary(dataset_factory, model, modality, loader, loss_fn, survival_train=None):
    r"""
    Run a validation loop on the trained model 
    
    Args: 
        - dataset_factory : SurvivalDatasetFactory
        - model : Pytorch model
        - modality : String
        - loader : Pytorch loader
        - loss_fn : custom loss function clas
        - survival_train : np.array
    
    Returns:
        - patient_results : dictionary
        - c_index : Float
        - c_index_ipcw : Float
        - BS : List
        - IBS : Float
        - iauc : Float
        - total_loss : Float

    """
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    total_loss = 0.

    all_risk_scores = []
    all_risk_by_bin_scores = []
    all_censorships = []
    all_event_times = []
    all_clinical_data = []
    all_logits = []
    all_slide_ids = []

    slide_ids = loader.dataset.metadata['slide_id']
    count = 0
    with torch.no_grad():
        for data in loader:

            data_WSI, mask, y_disc, event_time, censor, data_omics, clinical_data_list, mask = _unpack_data(modality, device, data)

            if modality in ["coattn", "coattn_motcat"]:  
                h = model(
                    x_path=data_WSI, 
                    x_omic1=data_omics[0], 
                    x_omic2=data_omics[1], 
                    x_omic3=data_omics[2], 
                    x_omic4=data_omics[3], 
                    x_omic5=data_omics[4], 
                    x_omic6=data_omics[5]
                )  
            elif modality in ("snn_chebykan_snn_dual_attn_lora",
                              "snn_chebykan_snn_dual_attn_lora_wo_residual",
                              "snn_chebykan_snn_dual_attn_lora_wo_snn",
                              "snn_chebykan_snn_dual_attn_lora_wo_chebykan",
                              "snn_chebykan_snn_dual_attn_lora_wo_attn",
                               "snn_chebykan_snn_dual_attn_lora_wo_omics",
                               "snn_chebykan_snn_dual_attn_lora_wo_wsi",
                               "snn_chebykan_snn_dual_attn_lora_wo_biattn",
                               "snn_chebykan_snn_dual_attn_lora_wo_lora",):

                input_args = {"x_path": data_WSI.to(device)}
                for i in range(len(data_omics)):
                    input_args['x_omic%s' % str(i+1)] = data_omics[i].type(torch.FloatTensor).to(device)
                input_args["return_attn"] = False
                
                h = model(**input_args)  
                
            else:
                h = model(
                    data_omics = data_omics, 
                    data_WSI = data_WSI, 
                    mask = mask
                    )
                    
            if len(h.shape) == 1:
                h = h.unsqueeze(0)
            loss = loss_fn(h=h, y=y_disc, t=event_time, c=censor)
            loss_value = loss.item()
            loss = loss / y_disc.shape[0]

            risk, risk_by_bin = _calculate_risk(h) # risk, survival
            all_risk_by_bin_scores.append(risk_by_bin)
            all_risk_scores, all_censorships, all_event_times, clinical_data_list = _update_arrays(all_risk_scores, all_censorships, all_event_times,all_clinical_data, event_time, censor, risk, clinical_data_list)
            all_logits.append(h.detach().cpu().numpy())
            total_loss += loss_value
            all_slide_ids.append(slide_ids.values[count])
            count += 1

    total_loss /= len(loader.dataset)
    all_risk_scores = np.concatenate(all_risk_scores, axis=0)
    all_risk_by_bin_scores = np.concatenate(all_risk_by_bin_scores, axis=0)
    all_censorships = np.concatenate(all_censorships, axis=0)
    all_event_times = np.concatenate(all_event_times, axis=0)
    all_logits = np.concatenate(all_logits, axis=0)
    
    patient_results = {}
    for i in range(len(all_slide_ids)):
        slide_id = slide_ids.values[i]
        case_id = slide_id[:12]
        patient_results[case_id] = {}
        patient_results[case_id]["time"] = all_event_times[i]
        patient_results[case_id]["risk"] = all_risk_scores[i]
        patient_results[case_id]["censorship"] = all_censorships[i]
        patient_results[case_id]["clinical"] = all_clinical_data[i]
        patient_results[case_id]["logits"] = all_logits[i]
    
    c_index, c_index2, BS, IBS, iauc = _calculate_metrics(loader, dataset_factory, survival_train, all_risk_scores, all_censorships, all_event_times, all_risk_by_bin_scores)
    
    return patient_results, c_index, c_index2, BS, IBS, iauc, total_loss


def _get_lr_scheduler(args, optimizer, dataloader):
    scheduler_name = args.lr_scheduler
    warmup_epochs = args.warmup_epochs
    epochs = args.max_epochs if hasattr(args, 'max_epochs') else args.epochs

    if warmup_epochs > 0:
        warmup_steps = warmup_epochs * len(dataloader)
    else:
        warmup_steps = 0
    if scheduler_name=='constant':
        lr_scheduler = get_constant_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=warmup_steps
        )
    elif scheduler_name=='cosine':
        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=len(dataloader) * epochs,
        )
    elif scheduler_name=='linear':
        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=len(dataloader) * epochs,
        )
    return lr_scheduler

fold_results_dict = {}

def _step(cur, args, loss_fn, model, optimizer, scheduler, train_loader, val_loader):
    r"""
    Trains the model for the set number of epochs and validates it.
    
    Args:
        - cur
        - args
        - loss_fn
        - model
        - optimizer
        - lr scheduler 
        - train_loader
        - val_loader
        
    Returns:
        - results_dict : dictionary
        - val_cindex : Float
        - val_cindex_ipcw  : Float
        - val_BS : List
        - val_IBS : Float
        - val_iauc : Float
        - total_loss : Float
    """
   

    all_survival = _extract_survival_metadata(train_loader, val_loader)

    best_cindex = 0
    best_epoch = -1
    best_ckpt_path = os.path.join(args.results_dir, f"s_{cur}_checkpoint.pt")

    
    for epoch in range(args.max_epochs):
        _train_loop_survival(epoch, model, args.modality, train_loader, optimizer, scheduler, loss_fn)
        _, val_cindex, _, _, _, _, total_loss = _summary(args.dataset_factory, model, args.modality, val_loader, loss_fn, all_survival)
        print('Val loss:', total_loss, ', val_c_index:', val_cindex)

        if val_cindex > best_cindex:
            best_cindex = val_cindex
            best_epoch = epoch
            torch.save(model.state_dict(), best_ckpt_path)
            print(f"New best model saved at epoch {epoch} with c-index {val_cindex:.4f}")

    model.load_state_dict(torch.load(best_ckpt_path))
    
    results_dict, val_cindex, val_cindex_ipcw, val_BS, val_IBS, val_iauc, total_loss = _summary(args.dataset_factory, model, args.modality, val_loader, loss_fn, all_survival)

    
    # Save intermediate results for current fold
    if hasattr(model, 'intermediate_results'):
        fold_results_dict[cur] = model.intermediate_results
        torch.save(model.intermediate_results, os.path.join(args.results_dir, f"intermediate_results_fold_{cur}.pth"))

    print('Final Val c-index: {:.4f}'.format(val_cindex))

    return results_dict, (val_cindex, val_cindex_ipcw, val_BS, val_IBS, val_iauc, total_loss)

def _train_val(datasets, cur, args):
    """   
    Performs train val test for the fold over number of epochs

    Args:
        - datasets : tuple
        - cur : Int 
        - args : argspace.Namespace 
    
    Returns:
        - results_dict : dict
        - val_cindex : Float
        - val_cindex2 : Float
        - val_BS : Float
        - val_IBS : Float
        - val_iauc : Float
        - total_loss : Float
    """

    #----> gets splits and summarize
    train_split, val_split = _get_splits(datasets, cur, args)

    #----> init model
    model = _init_model(args)
    
    #----> init loss function
    loss_fn = _init_loss_function(args)
    
    #---> init optimizer
    optimizer = _init_optim(args, model)

    #---> init loaders
    train_loader, val_loader = _init_loaders(args, train_split, val_split)

    # lr scheduler 
    lr_scheduler = _get_lr_scheduler(args, optimizer, train_loader)

    #---> do train val
    results_dict, (val_cindex, val_cindex2, val_BS, val_IBS, val_iauc, total_loss) = _step(cur, args, loss_fn, model, optimizer, lr_scheduler, train_loader, val_loader)

    return results_dict, (val_cindex, val_cindex2, val_BS, val_IBS, val_iauc, total_loss)

