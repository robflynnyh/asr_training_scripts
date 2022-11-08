echo '<<< TRAINING AUTOREGRESSIVE TLM >>>'

python train_LM.py --checkpoint '' \
    --checkpoint_dir './checkpoints/checkpoint_lm_ami_test_8000_90s' \
    --model_config './experiment_configs/lm/decoder_test.yaml' \
    --min_lr 1e-5 \
    --max_lr 3.7e-4 \
    --step_size 70 \
    --accumulate_gradients 6 \
    --clip_gradients \
    --clip_gradients_value 15 \
    --micro_batch_duration 90 \
    --micro_batch_number 5 \
    --max_allowed_utterance_gap 3.0 \
    --wandb_id '' \
    --save_top_k 1 \
    --schedular_data './checkpoints/ami_test_8000_90s.json' \
    --project_name 'AMI_lms' \
    



#GCC/9.3.0
#    --wandb_id '3bc1yvgb' \

#checkpoint_26_id_78.pt

# step_size = step size up, step size down is step_size*4 
