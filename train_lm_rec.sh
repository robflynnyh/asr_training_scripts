echo '<<< TRAINING AUTOREGRESSIVE TLM >>>'

python rec_rescoring.py --checkpoint 'checkpoint_316_id_46_opensub_unigram.pt' \
    --train_hyp 'train_ami.pkl' \
    --dev_hyp 'dev_ami.pkl' \
    --checkpoint_dir './checkpoints/open_sub_ft_ami/' \
    --config './experiment_configs/lm/decoder_pg19_sep_token_ami_vocab.yaml' \
    --min_lr 1e-6 \
    --max_lr 5e-5 \
    --step_size 35 \
    --clip_gradients \
    --clip_gradients_value 15 \
    --max_allowed_utterance_gap 10.0 \
    --wandb_id '' \
    --save_top_k 1 \
    --schedular_data './pg191kwftpd90.json' \
    --utts 30 \
    -batch 10  \

    



#GCC/9.3.0
#    --wandb_id '3bc1yvgb' \

#checkpoint_26_id_78.pt

# step_size = step size up, step size down is step_size*4 
