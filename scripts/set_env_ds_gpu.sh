#set -e
#set -x

######################################################################
export DISTRIBUTED_BACKEND="nccl"
export CUDA_DEVICE_MAX_CONNECTIONS=1
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

######################################################################
python -m pip install -r requirements_ds_gpu.txt
python -m pip install -e `pwd`

######################################################################

export NNODES=${WORLD_SIZE}
export NODE_RANK=${RANK}
export MASTER_PORT=45678
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"

if [ -z "$NPROC_PER_NODE" ]
then
    export NPROC_PER_NODE=8
    export NNODES=1
    export NODE_RANK=0
    export MASTER_ADDR=127.0.0.1
fi

######################################################################
