echo '<<< TRAINING SMALL CTC-XL MODEL >>>'

python train_block_reccurrent.py --checkpoint '' \
    --checkpoint_dir './checkpoints_conformer_xl_test' \
    --model_config '../model_configs/rnn_conformer_sc_ctc_bpe_small.yaml' \
    --min_lr 3e-6 \
    --max_lr 7.0e-5 \
    --step_size 850 \
    --clip_gradients \
    --clip_gradients_value 15 \
    --micro_batch_duration 0 \
    --utts_per_micro_batch 1 \
    --micro_batch_number 20 \
    --epochs 80 \
    --wandb_id '' \
    --schedular_data 'prerunxl.json' \
    



#GCC/9.3.0
#    --wandb_id '3bc1yvgb' \

#checkpoint_26_id_78.pt

# step_size = step size up, step size down is step_size*4 
