echo '<<< TRAINING SMALL CTC MODEL >>>'

python train_H.py --checkpoint '' \
    --checkpoint_dir './checkpoints_cosine_th_both_16h' \
    --model_config '../model_configs/conformer_sc_ctc_bpe_small.yaml' \
    --min_lr 4e-6 \
    --max_lr 1e-4 \
    --step_size 150 \
    --accumulate_gradients 1 \
    --clip_gradients \
    --clip_gradients_value 15 \
    --micro_batch_duration 0 \
    --micro_batch_number 20 \
    --do_not_pass_segment_lens \
    --wandb_id '' \
    --schedular_data 'cosine_dh4.json' \
    



#GCC/9.3.0
#    --wandb_id '3bc1yvgb' \

#checkpoint_26_id_78.pt

# step_size = step size up, step size down is step_size*4 
