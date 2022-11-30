echo '<<< TRAINING AUTOREGRESSIVE TLM >>>'

python train_LM.py --checkpoint '' \
    --checkpoint_dir './checkpoints/checkpoints_3waypos2' \
    --model_config './experiment_configs/lm/decoder_test.yaml' \
    --min_lr 5e-5 \
    --max_lr 3.5e-4 \
    --step_size 300 \
    --accumulate_gradients 5\
    --clip_gradients \
    --clip_gradients_value 15 \
    --micro_batch_duration 90 \
    --micro_batch_number 6 \
    --max_allowed_utterance_gap 10.0 \
    --wandb_id '' \
    --save_top_k 1 \
    --schedular_data './checkpoints/3waypos3.json' \
    --project_name 'deliberation-LM' \
    --label_smoothing 0.0 \
    



#GCC/9.3.0
#    --wandb_id '3bc1yvgb' \

#checkpoint_26_id_78.pt

# step_size = step size up, step size down is step_size*4 
