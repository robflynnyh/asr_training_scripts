echo '<<< TRAINING SMALL CTC MODEL >>>'

python train_H.py --checkpoint '' \
    --checkpoint_dir './checkpoints_cosine_nths' \
    --model_config '../model_configs/conformer_sc_ctc_bpe_small.yaml' \
    --min_lr 1e-5 \
    --max_lr 1.3e-4 \
    --step_size 70 \
    --accumulate_gradients 4 \
    --clip_gradients \
    --clip_gradients_value 15 \
    --micro_batch_duration 0 \
    --micro_batch_number 25 \
    --do_not_pass_segment_lens \
    --wandb_id '' \
    --schedular_data 'cosineted_nths.json' \
    



#GCC/9.3.0
#    --wandb_id '3bc1yvgb' \

#checkpoint_26_id_78.pt

# step_size = step size up, step size down is step_size*4 
