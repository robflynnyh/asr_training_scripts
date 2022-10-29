echo '<<< TRAINING SMALL CTC MODEL >>>'

python train.py --checkpoint '' \
    --checkpoint_dir './checkpoints' \
    --model_config '/exp/exp1/acp21rjf/deliberation/Custom/model_configs/conformer_ctc_bpe_small.yaml' \
    --min_lr 1e-5 \
    --max_lr 3e-4 \
    --step_size 300


# step_size = step size up, step size down is step_size*4 