export CUDA_VISIBLE_DEVICES=$1
echo ${CUDA_VISIBLE_DEVICES}
local_and_nonlocal=mmc
plm=google/electra-large-discriminator
# google/electra-large-discriminator
# microsoft/deberta-large
# plm=microsoft/deberta-v3-large
layer=16
global_drop_edge_p=0.0
local_drop_edge_p=0.0

folder='.'
# folder=exp/task_lgesql_large__model_lgesql_view_${local_and_nonlocal}_gp_0.15/plm_${plm}__gnn_512_x_${layer}__share__bs_20__bm_5__seed_999__local_d_edge_${local_drop_edge_p}__global_d_edge_${global_drop_edge_p}
distributed=no
max_epoch=200

bash ./run/run_lgesql_plm.sh ${folder} ${local_and_nonlocal} ${plm} ${layer} ${global_drop_edge_p} ${local_drop_edge_p} ${max_epoch} ${distributed}