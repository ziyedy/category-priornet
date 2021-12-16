if [ "$1" = "3dgcn" ]; then
    echo "------------- train 3dgcn ---------------"
    python3 main.py \
    -mode train \
    -support 1 \
    -neighbor 20 \
    -cuda 0 \
    -epoch 100 \
    -bs 8 \
    -dataset /datasets/DATASET/ModelNet40/Dataset \
    -record ./results/3dgcn-record.log \
    -info ./results/3dgcn \
    -save ./results/3dgcn-model.pkl \
    #-normal
elif [ "$1" = "pointnet" ]; then
    echo "------------- train pointnet ---------------"
    python3 main.py \
    -mode train \
    -model pointnet \
    -support 1 \
    -neighbor 20 \
    -cuda 0 \
    -epoch 100 \
    -bs 8 \
    -dataset /datasets/DATASET/ModelNet40/Dataset \
    -record ./results/pointnet-record.log \
    -info ./results/pointnet \
    -save ./results/pointnet-model.pkl \
    -normal
elif [ "$1" = "dgcnn" ]; then
    echo "------------- train dgcnn ---------------"
    python3 main.py \
    -mode train \
    -model dgcnn \
    -support 1 \
    -neighbor 20 \
    -cuda 0 \
    -epoch 100 \
    -bs 8 \
    -dataset /datasets/DATASET/ModelNet40/Dataset \
    -record ./results/dgcnn-record.log \
    -info ./results/dgcnn \
    -save ./results/dgcnn-model.pkl \
    -normal \
#    -rotate 180 \
#    -random
fi
