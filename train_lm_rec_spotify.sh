echo '<<< TRAINING AUTOREGRESSIVE TLM >>>'

python rec_rescoring_opensub_v2.py --checkpoint 'checkpoint_316_id_46_opensub_unigram.pt' \
    --checkpoint_dir './checkpoints/open_sub_ft_ami/' \
    --config './experiment_configs/lm/decoder_pg19_sep_token_ami_vocab.yaml' \
    --min_lr 1e-6 \
    --max_lr 2.5e-4 \
    --step_size 250 \
    --clip_gradients \
    --clip_gradients_value 15 \
    --max_allowed_utterance_gap 10.0 \
    --wandb_id '' \
    --save_top_k 15 \
    --schedular_data './pg191kwftpd90.json' \
    --utts 25 \
    -batch 30  \

    



#GCC/9.3.0
#    --wandb_id '3bc1yvgb' \

#checkpoint_26_id_78.pt

# step_size = step size up, step size down is step_size*4 
