echo '<<< TRAINING AUTOREGRESSIVE TLM >>>'

python rec_rescoring.py --checkpoint '' \
    --train_hyp 'train_ted_lm.pkl' \
    --dev_hyp 'dev_ted_lm.pkl' \
    --checkpoint_dir './checkpoints/pos_test/' \
    --config './experiment_configs/lm/decoder_pg19_sep_token_ted_am.yaml' \
    --min_lr 2e-5 \
    --max_lr 1.155e-4 \
    --step_size 35 \
    --clip_gradients \
    --clip_gradients_value 15 \
    --max_allowed_utterance_gap 10.0 \
    --wandb_id '' \
    --save_top_k 1 \
    --schedular_data './pos_test.json' \
    --utts 16 \
    -batch 15  \

    



#GCC/9.3.0
#    --wandb_id '3bc1yvgb' \

#checkpoint_26_id_78.pt

# step_size = step size up, step size down is step_size*4 
