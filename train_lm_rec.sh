echo '<<< TRAINING AUTOREGRESSIVE TLM >>>'

python rec_rescoring.py --checkpoint 'checkpoint_182_id_61.pt' \
    --train_hyp 'train.pkl' \
    --dev_hyp 'dev.pkl' \
    --checkpoint_dir './checkpoints/test_open_sub_sep/' \
    --config './experiment_configs/lm/decoder_pg19_sep_token.yaml' \
    --min_lr 1e-6 \
    --max_lr 8e-5 \
    --step_size 100 \
    --clip_gradients \
    --clip_gradients_value 15 \
    --max_allowed_utterance_gap 10.0 \
    --wandb_id '' \
    --save_top_k 1 \
    --schedular_data './pg191kwftpd90.json' \
    --utts 25 \
    -batch 80  \

    



#GCC/9.3.0
#    --wandb_id '3bc1yvgb' \

#checkpoint_26_id_78.pt

# step_size = step size up, step size down is step_size*4 
