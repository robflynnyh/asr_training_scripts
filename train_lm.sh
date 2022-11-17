echo '<<< TRAINING AUTOREGRESSIVE TLM >>>'

python train_LM.py --checkpoint '' \
    --checkpoint_dir './checkpoints/checkpoint_tedlium_90_29_50do_shgau' \
    --model_config './experiment_configs/lm/decoder_test.yaml' \
    --min_lr 3.5e-5 \
    --max_lr 4.5e-4 \
    --step_size 100 \
    --accumulate_gradients 8\
    --clip_gradients \
    --clip_gradients_value 15 \
    --micro_batch_duration 90 \
    --micro_batch_number 3 \
    --max_allowed_utterance_gap 3.0 \
    --wandb_id '' \
    --save_top_k 1 \
    --schedular_data './checkpoints/tedlium_90_29_50do_shgau.json' \
    --project_name 'deliberation-LM' \
    



#GCC/9.3.0
#    --wandb_id '3bc1yvgb' \

#checkpoint_26_id_78.pt

# step_size = step size up, step size down is step_size*4 
