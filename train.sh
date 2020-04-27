python3 main.py -data_dir data -hidden_dim 256  -embed_dim 40 \
-batch_size 64 -epochs 300 -lr 3e-4 -clip_value 1.0 -w_decay 0.00001 \
-max_decoding_length 300 -is_stochastic 1 -train 1 -models_dir models \
-logs_dir logs -model_path best.pth -num_workers 64 -tf 0.3 -locked_dropout 0.9