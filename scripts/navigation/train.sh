export CUDA_VISIBLE_DEVICES=0

python src/train.py \
    --name 0200200 \
    --task nav \
    --label-type color
    --plm bert-small-uncased \
    --prefix-hidden-size 256
