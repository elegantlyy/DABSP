#!/bin/bash
DATA_ROOT_DIR="./features" # where are the TCGA features stored?
TYPE_OF_PATH="combine" # what type of pathways? 
MODEL="snn_chebykan_snn_dual_attn_lora" # what type of model do you want to train?
DIM1=8
DIM2=16
STUDIES=("coad")
LRS=0.00005
DECAYS=0.00001
CHEBY_HIDDEN=2
DEGREE=6
SURV_WEIGHTS=0.5
EPOCHS=60
LORAR=32
LORAALPHA=32
NUMBER_GENES_PER_PATHWAY=100

for STUDY in ${STUDIES[@]};
do
    echo "Running with lora_r=$LORAR and lora_alpha=$LORAALPHA"
    CUDA_VISIBLE_DEVICES=1 python main.py \
    --study tcga_${STUDY} --task survival --split_dir splits --which_splits 5foldcv \
    --type_of_path $TYPE_OF_PATH --modality $MODEL --data_root_dir ${DATA_ROOT_DIR}/tcga_${STUDY}/ \
    --label_file datasets_csv/metadata/tcga_${STUDY}.csv  \
    --omics_dir datasets_csv/raw_rna_data/${TYPE_OF_PATH}/${STUDY} \
    --lora_r $LORAR --lora_alpha $LORAALPHA \
    --results_dir results_${STUDY} \
    --batch_size 1 --lr $LRS --opt radam --reg $DECAYS --cheby_hidden $CHEBY_HIDDEN \
    --cheby_degree $DEGREE --per_pathway_over_n $number \
    --alpha_surv $SURV_WEIGHTS --weighted_sample --max_epochs $EPOCHS --encoding_dim 1024 \
    --label_col survival_months_dss --k 5 --bag_loss nll_surv --n_classes 4 --num_patches 4096 --wsi_projection_dim 384 \
    --encoding_layer_1_dim ${DIM1} --encoding_layer_2_dim ${DIM2} --encoder_dropout 0.25 
done