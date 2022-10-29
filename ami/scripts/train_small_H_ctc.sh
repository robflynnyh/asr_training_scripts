echo '<<< TRAINING SMALL CROSS-CONTEXT CTC MODEL >>>'

python train_H.py --checkpoint '' \
    --checkpoint_dir './checkpoints' \
    --model_config '/exp/exp1/acp21rjf/deliberation/Custom/model_configs/Hconformer_sc_ctc_bpe_small.yaml' \
    --min_lr 1e-5 \
    --max_lr 1.25e-4 \
    --step_size 1600 \
    --accumulate_gradients 5 \
    --clip_gradients \
    --clip_gradients_value 10 \
    --micro_batch_size 6 \
    --micro_batch_range 1 \
    --micro_batch_number 5 \
    --schedular_data 'Hconf_sched_sccrosscontext.json' \




#    --wandb_id '3bc1yvgb' \

#checkpoint_26_id_78.pt

# step_size = step size up, step size down is step_size*4 
