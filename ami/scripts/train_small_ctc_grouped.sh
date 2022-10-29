echo '<<< TRAINING SMALL GROUPED CTC MODEL >>>'

python train.py --checkpoint '' \
    --checkpoint_dir './checkpoints_grouped' \
    --model_config '/exp/exp1/acp21rjf/deliberation/Custom/model_configs/conformer_grouped_ctc_bpe_small.yaml' \
    --min_lr 1e-5 \
    --max_lr 8e-5 \
    --step_size 1125 \
    --accumulate_gradients 2 \
    --schedular_data 'schedular_grouped_attn_ctc.json' \
    --batch_length 450 \


# step_size = step size up, step size down is step_size*4 