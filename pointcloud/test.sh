if [ "$1" = "3dgcn" ]; then
    echo "------------- test 3dgcn ---------------"
    python3 main.py \
    -mode test \
    -cuda 0 \
    -bs 8 \
    -dataset /datasets/DATASET/ModelNet40/Dataset \
    -support 1 \
    -neighbor 20 \
    -load ./results/3dgcn-model.pkl \
    -random \
    -rotate 180 \
    -scale 1.8 \
    -shift 5.0
    #-normal \
elif [ "$1" = "pointnet" ]; then
    echo "------------- test pointnet ---------------"
    python3 main.py \
    -mode test \
    -model pointnet \
    -cuda 0 \
    -bs 8 \
    -dataset /datasets/DATASET/ModelNet40/Dataset \
    -support 1 \
    -neighbor 20 \
    -load ./results/pointnet-model.pkl \
    -random \
    -rotate 180 \
    -scale 1.8 \
    -shift 5.0
    #-normal \
elif [ "$1" = "dgcnn" ]; then
    echo "------------- test dgcnn ---------------"
    python3 main.py \
    -mode test \
    -model dgcnn \
    -cuda 0 \
    -bs 8 \
    -dataset /datasets/DATASET/ModelNet40/Dataset \
    -support 1 \
    -neighbor 20 \
    -load ./results/dgcnn-model.pkl \
    -random \
    -rotate 180 \
    -scale 1.8 \
    -shift 5.0
    #-normal \
fi
