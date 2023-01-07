echo '<<< TRAINING AUTOREGRESSIVE TLM >>>'

python train_LM.py --checkpoint '' \
    --checkpoint_dir './checkpoints/t0_b15/' \
    --model_config './experiment_configs/lm/decoder_test.yaml' \
    --min_lr 1e-6 \
    --max_lr 1e-4 \
    --step_size 800 \
    --accumulate_gradients 1 \
    --clip_gradients \
    --clip_gradients_value 15 \
    --micro_batch_duration 0 \
    --micro_batch_number 15 \
    --max_allowed_utterance_gap 10.0 \
    --wandb_id '' \
    --save_top_k 1 \
    --schedular_data './t0b15.json' \
    --project_name 'LM-context_len_stability_tests' \
    --label_smoothing 0.0 \
    



#GCC/9.3.0
#    --wandb_id '3bc1yvgb' \

#checkpoint_26_id_78.pt

# step_size = step size up, step size down is step_size*4 
