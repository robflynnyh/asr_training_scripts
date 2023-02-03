echo '<<< TRAINING AUTOREGRESSIVE TLM >>>'

python rec_rescoring.py --checkpoint 'pg_19_pretrained_c_62_id_87.pt' \
    --train_hyp 'train.pkl' \
    --dev_hyp 'dev.pkl' \
    --checkpoint_dir './checkpoints/1kw_pg19checkpoints_nths/' \
    --config './experiment_configs/lm/decoder_pg19.yaml' \
    --min_lr 3e-6 \
    --max_lr 4.5e-5 \
    --step_size 250 \
    --accumulate_gradients 1 \
    --clip_gradients \
    --clip_gradients_value 15 \
    --max_allowed_utterance_gap 10.0 \
    --wandb_id '' \
    --save_top_k 1 \
    --schedular_data './pg191kwftpd90.json' \
    --utts 10 \
    -batch 30  \

    



#GCC/9.3.0
#    --wandb_id '3bc1yvgb' \

#checkpoint_26_id_78.pt

# step_size = step size up, step size down is step_size*4 
