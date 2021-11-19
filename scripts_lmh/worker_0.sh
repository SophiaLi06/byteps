export NVIDIA_VISIBLE_DEVICES=0,1,2,3
export DMLC_WORKER_ID=0
export DMLC_NUM_WORKER=2
export DMLC_ROLE=worker
export DMLC_NUM_SERVER=1
export DMLC_PS_ROOT_URI=10.31.179.$1  # the scheduler IP
export DMLC_PS_ROOT_PORT=1234 # the scheduler port