echo '<<< TRAINING SMALL CTC MODEL >>>'

python train_H.py --checkpoint '' \
    --checkpoint_dir './checkpoints/earnings_run' \
    --model_config '../model_configs/conformer_sc_ctc_bpe_small.yaml' \
    --min_lr 1e-6 \
    --max_lr 4e-5 \
    --step_size 250 \
    --accumulate_gradients 1 \
    --clip_gradients \
    --clip_gradients_value 15 \
    --micro_batch_duration 0 \
    --micro_batch_number 15 \
    --do_not_pass_segment_lens \
    --wandb_id '' \
    --schedular_data 'earnings_run.json' \
    



#GCC/9.3.0
#    --wandb_id '3bc1yvgb' \

#checkpoint_26_id_78.pt

# step_size = step size up, step size down is step_size*4 
