layer=32
bs=20
local_drop_edge_p=0.0
global_drop_edge_p=0.0

exp_name="exp/task_lgesql_large__model_lgesql_view_mmc_gp_0.15/plm_google/electra-large-discriminator__gnn_512_x_${layer}__share__bs_${bs}__bm_5__seed_999__local_d_edge_${local_drop_edge_p}__global_d_edge_${global_drop_edge_p}"

python3 my_scripts/evaluation_format.py --experiment_name=${exp_name}

