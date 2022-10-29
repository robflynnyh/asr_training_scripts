echo '<<< TRAINING SMALL CTC MODEL >>>'

python train_H.py --checkpoint '' \
    --checkpoint_dir './checkpoints' \
    --model_config '../model_configs/BERTconformer_sc_ctc_bpe_small.yaml' \
    --min_lr 1e-7 \
    --max_lr 1e-5 \
    --step_size 500 \
    --accumulate_gradients 4 \
    --clip_gradients \
    --clip_gradients_value 20 \
    --micro_batch_duration 0 \
    --micro_batch_number 5 \
    --do_not_pass_segment_lens \
    --wandb_id '' \
    --schedular_data 'TEDBERRT.json' \
    




#    --wandb_id '3bc1yvgb' \

#checkpoint_26_id_78.pt

# step_size = step size up, step size down is step_size*4 
