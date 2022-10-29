python train_sc_ctc.py \
    --checkpoint 'checkpoint_133_id_37.pt' \
    --model_config '/exp/exp1/acp21rjf/deliberation/Custom/model_configs/conformer_sc_ctc_bpe_small.yaml' \
    --min_lr 1e-5 \
    --max_lr 4.5e-4  \
    --step_size 800 \
    --wandb_id 'dir70kbq'