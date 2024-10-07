
# python -u train.py --dim_value 41 --num_books 64 --learning_rate 0.01 \
# --optimizer Adam  --hidden 16 --num_channels 16 --GFCIL high_resources \
# --pretrain_model_path /home/bqqi/graph-lifelong-learning/graph-lifelong-learning/graph_kvb/pretrain_model/Reddit/GCN/42_970.pth \
# --dataset Reddit --cl_epochs 500 > test_Reddit.log

# python -u train.py --dim_value 40 --num_books 64 --learning_rate 0.01 \
# --optimizer Adam  --hidden 128 --num_channels 16 --GFCIL high_resources \
# --pretrain_model_path ./pretrain_model/ogbn-arxiv/GCN/42_10.pth \
# --dataset ogbn-arxiv > test_Arxiv.log

# python -u test.py --dim_value 70 --num_books 64 --learning_rate 0.01 \
# --optimizer Adam  --hidden 128 --num_channels 128 --GFCIL high_resources \
# --pretrain_model_path ./pretrain_model/CoraFull/GCN/42_690.pth \
# --dataset CoraFull > test_CoraFull_now.log


############################# CS ##########################
python -u train_1.py --dim_value 70 --num_books 100 --learning_rate 0.0005 \
--optimizer Adam  --hidden 128 --num_channels 128 --GFCIL high_resources --dim_key 14 \
--pretrain_model_path ./pretrain_model/CoraFull/GCN/42_440.pth --dataset CoraFull --backbone GCN > test1.txt

