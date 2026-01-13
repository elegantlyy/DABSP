import argparse

def _process_args():
    r"""
    Function creates a namespace to read terminal-based arguments for running the experiment

    Args
        - None 

    Return:
        - args : argparse.Namespace

    """

    parser = argparse.ArgumentParser(description='Configurations for SurvPath Survival Prediction Training')

    #---> study related
    parser.add_argument('--study', type=str, help='study name')
    parser.add_argument('--task', type=str, choices=['survival'])
    parser.add_argument('--n_classes', type=int, default=4, help='number of classes (4 bins for survival)') 
    parser.add_argument('--results_dir', default='./results', help='results directory (default: ./results)')
    parser.add_argument("--type_of_path", type=str, default="hallmarks", choices=["xena", "hallmarks", "combine"]) 
    parser.add_argument('--testing', action='store_true', default=False, help='debugging tool')
    
    parser.add_argument('--per_pathway_over_n', type=int, default=0, help="per pathway over how many genes, can set 10,50,100,150,180,190..but max_gene=199")
    
    parser.add_argument('--kan_n', type=int, default=64, help='efficient-kan hidden layer dim')
    
    parser.add_argument('--cheby_hidden', type=int, default=64, help='cheby_kan-hidden layer dim')
    parser.add_argument('--cheby_degree', type=int, default=4, help='cheby_kan degree ')
    
    parser.add_argument('--fourier_hidden', type=int, default=64, help='fourier_kan-hidden layer dim')
    parser.add_argument('--gridsize', type=int, default=128, help='fourier_hidden gridsize')

    parser.add_argument('--lora_r', type=int, default=4, help='dual-attn, q, k, v')
    parser.add_argument('--lora_alpha', type=int, default=16, help='dual-attn, q, k, v')

    parser.add_argument('--our_fusion', type=str, default='concat', help='concat , kp')

    parser.add_argument('--wo_omics', type=int, default=0, help='ablation study, only_use_wsi')
    parser.add_argument('--wo_wsi', type=int, default=0, help='ablation study, only_use_omics')



    #----> data related
    parser.add_argument('--data_root_dir', type=str, default=None, help='data feature directory')
    parser.add_argument('--label_file', type=str, default=None, help='Path to csv with labels')
    parser.add_argument('--omics_dir', type=str, default=None, help='Path to dir with omics csv for all modalities') 
    parser.add_argument('--num_patches', type=int, default=4000, help='number of patches') # 4096
    parser.add_argument('--label_col', type=str, default="survival_months_dss", help='type of survival (OS, DSS, PFI)')
    parser.add_argument("--wsi_projection_dim", type=int, default=1) 
    parser.add_argument("--encoding_layer_1_dim", type=int, default=8)
    parser.add_argument("--encoding_layer_2_dim", type=int, default=16)
    parser.add_argument("--encoder_dropout", type=float, default=0.25)


    #----> split related 
    parser.add_argument('--k', type=int, default=5, help='number of folds (default: 10)')
    parser.add_argument('--k_start', type=int, default=-1, help='start fold (default: -1, last fold)')
    parser.add_argument('--k_end', type=int, default=-1, help='end fold (default: -1, first fold)')
    parser.add_argument('--split_dir', type=str, default=None, help='manually specify the set of splits to use, ' 
                    +'instead of infering from the task and label_frac argument (default: None)')
    parser.add_argument('--which_splits', type=str, default="10foldcv", help='where are splits') 
        
    #----> training related 
    parser.add_argument('--max_epochs', type=int, default=20, help='maximum number of epochs to train (default: 200)') # 2
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate (default: 0.0001)')
    parser.add_argument('--seed', type=int, default=1, help='random seed for reproducible experiment (default: 1)')
    parser.add_argument('--opt', type=str, default="adam", help="Optimizer")
    parser.add_argument('--reg_type', type=str, default="None", help="regularization type [None, L1, L2]") # l2
    parser.add_argument('--weighted_sample', action='store_true', default=False, help='enable weighted sampling')
    parser.add_argument('--batch_size', type=int, default=1, help='batch_size')
    
    parser.add_argument('--bag_loss', type=str, choices=['ce_surv', "nll_surv", "nll_rank_surv", "rank_surv", "cox_surv", "nll_surv_mse", "nll_ortho_surv"], default='ce',
                        help='survival loss function (default: ce)')
    parser.add_argument('--nll_weight', type=float, default=1.0, help='weight given to raw NLL loss') # for MAE
    parser.add_argument('--mae_weight', type=float, default=0.5, help='weight given to MAE loss') # for MAE

    parser.add_argument('--alpha_surv', type=float, default=0.0, help='weight given to uncensored patients') 
    parser.add_argument('--reg', type=float, default=1e-5, help='weight decay / L2 (default: 1e-5)')
    parser.add_argument('--lr_scheduler', type=str, default='cosine')
    parser.add_argument('--warmup_epochs', type=int, default=1)
    
    
    

    #---> model related
    parser.add_argument('--fusion', type=str, default=None)
    parser.add_argument('--modality', type=str, default="survpath", help="wsi, mcat, abmil,,,survpath_ctranspath,etc")
    parser.add_argument('--encoding_dim', type=int, default=768, help='WSI encoding dim')
    parser.add_argument('--use_nystrom', action='store_true', default=False, help='Use Nystrom attentin in SurvPath.')

    args = parser.parse_args()

    if not (args.task == "survival"):
        print("Task and folder does not match")
        exit()

    return args