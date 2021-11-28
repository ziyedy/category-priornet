if [ "$1" = "3dgcn" ]; then
    echo "------------- train 3dgcn ---------------"
    python3 main.py \
    -mode train \
    -support 1 \
    -neighbor 20 \
    -cuda 0 \
    -epoch 100 \
    -bs 8 \
    -dataset ./Dataset \
    -record ./results/3dgcn-record.log \
    -save ./results/3dgcn-model.pkl \
    #-normal
elif [ "$1" = "pointnet" ]; then
    echo "------------- train pointnet ---------------"
    python3 main.py \
    -mode train \
    -support 1 \
    -neighbor 20 \
    -cuda 0 \
    -epoch 100 \
    -bs 8 \
    -dataset ./Dataset \
    -record ./results/pointnet-record.log \
    -save ./results/pointnet-model.pkl \
    -normal
elif [ "$1" = "dgcnn" ]; then
    echo "------------- train dgcnn ---------------"
    python3 main.py \
    -mode train \
    -support 1 \
    -neighbor 20 \
    -cuda 0 \
    -epoch 100 \
    -bs 8 \
    -dataset ./Dataset \
    -record ./results/dgcnn-record.log \
    -save ./results/dgcnn-model.pkl \
    -normal
fi
