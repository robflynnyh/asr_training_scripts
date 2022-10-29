echo '<<< TRAINING SMALL CHAR CTC MODEL >>>'

python train.py --checkpoint 'checkpoint_328_id_26.pt' \
    --checkpoint_dir './checkpoints' \
    --model_config '/exp/exp1/acp21rjf/deliberation/Custom/model_configs/conformer_ctc_char_small.yaml' \
    --min_lr 1e-5 \
    --max_lr 5e-4 \
    --step_size 450 \
    --accumulate_gradients 16 \
    --tokenizer 'tokenizer_spe_char' \
    --schedular_data 'char_level_ctc_small.json' \
    --wandb_id 'vfccke3f' \
    --epochs 1000

echo "<<< OKAY, WE'RE DONE HERE >>>"

# step_size = step size up, step size down is step_size*4 