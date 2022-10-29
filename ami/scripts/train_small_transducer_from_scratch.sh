echo '<<< TRAINING SMALL TRANDUCER MODEL >>>'

python train_transducer.py \
    --checkpoint '' \
    --model_config '/exp/exp1/acp21rjf/deliberation/Custom/model_configs/conformer_transducer_bpe_small.yaml' \
    --min_lr 1e-5 \
    --max_lr 6e-4  \
    --step_size 800 

echo '<<< OKAY BYE >>>'