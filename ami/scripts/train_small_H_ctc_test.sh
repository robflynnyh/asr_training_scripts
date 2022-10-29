echo '<<< TRAINING SMALL CROSS-CONTEXT CTC MODEL >>>'

python train.py --checkpoint '' \
    --checkpoint_dir './checkpoints' \
    --model_config '/exp/exp1/acp21rjf/deliberation/Custom/model_configs/Hconformer_ctc_bpe_small.yaml' \
    --min_lr 1e-5 \
    --max_lr 2e-4 \
    --step_size 1200 \
    --accumulate_gradients 1 \







#checkpoint_26_id_78.pt

# step_size = step size up, step size down is step_size*4 
