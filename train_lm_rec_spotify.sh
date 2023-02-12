echo '<<< TRAINING AUTOREGRESSIVE TLM >>>'

python rec_rescoring_opensub.py --checkpoint 'pg_19_pretrained_c_62_id_87.pt' \
    --checkpoint_dir './checkpoints/test_open_sub_sep/' \
    --config './experiment_configs/lm/decoder_pg19_sep_token.yaml' \
    --min_lr 4e-5 \
    --max_lr 3.0e-4 \
    --step_size 250 \
    --clip_gradients \
    --clip_gradients_value 15 \
    --max_allowed_utterance_gap 10.0 \
    --wandb_id '' \
    --save_top_k 15 \
    --schedular_data './pg191kwftpd90.json' \
    --utts 15 \
    -batch 25  \

    



#GCC/9.3.0
#    --wandb_id '3bc1yvgb' \

#checkpoint_26_id_78.pt

# step_size = step size up, step size down is step_size*4 
