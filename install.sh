# pip install
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.11.0+cu115.html
pip install transformers
pip install tensorflow
pip install tensorboard
pip install protobuf==3.19.4
pip install gdown

sudo apt update
sudo apt -y install wget
sudo apt -y install unzip

if [ ! -d ./data ]; then
    mkdir ./data
fi

# download CLEVR dataset
if [ ! -f ./data/CLEVR_train_scenes.json ] || [ ! -f ./data/CLEVR_val_scenes.json ]; then
    wget https://dl.fbaipublicfiles.com/clevr/CLEVR_v1.0_no_images.zip
    unzip CLEVR_v1.0_no_images.zip
    if [ ! -f ./data/CLEVR_train_scenes.json ]; then
        cp CLEVR_v1.0/scenes/CLEVR_train_scenes.json ./data/
    fi
    if [ ! -f ./data/CLEVR_val_scenes.json ]; then
        cp CLEVR_v1.0/scenes/CLEVR_val_scenes.json ./data/
    fi
    rm CLEVR_v1.0_no_images.zip
    rm -r CLEVR_v1.0
fi

# download common sense corpus from Google Drive
if [ ! -f ./data/common_sense_corpus.csv ]; then
    gdown https://drive.google.com/uc?id=1tyQWKA_-iZHLy0hULy749YntJXgSbRwJ
    cp common_sense_corpus.csv ./data
    rm common_sense_corpus.csv
fi
