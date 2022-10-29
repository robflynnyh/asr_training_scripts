echo '<<< TRAINING SMALL CTC MODEL >>>'

python train_H.py --checkpoint 'checkpoint_101_id_82.pt' \
    --checkpoint_dir './checkpoints_flash' \
    --model_config '../model_configs/conformer_sc_ctc_bpe_small.yaml' \
    --min_lr 1e-7 \
    --max_lr 5e-5 \
    --step_size 500 \
    --accumulate_gradients 2 \
    --clip_gradients \
    --clip_gradients_value 10 \
    --micro_batch_duration 0 \
    --gap 0.1 \
    --micro_batch_number 30 \
    --do_not_pass_segment_lens \
    --wandb_id '2csgaq4s' \
    --schedular_data 'TEDFLASH.json' \
    



#GCC/9.3.0
#    --wandb_id '3bc1yvgb' \

#checkpoint_26_id_78.pt

# step_size = step size up, step size down is step_size*4 
