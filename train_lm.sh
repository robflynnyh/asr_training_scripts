echo '<<< TRAINING AUTOREGRESSIVE TLM >>>'

python train_LM.py --checkpoint '' \
    --checkpoint_dir './checkpoints/lm_test/' \
    --model_config './experiment_configs/lm/decoder_test.yaml' \
    --min_lr 4e-5 \
    --max_lr 3.7e-4 \
    --step_size 100 \
    --accumulate_gradients 10\
    --clip_gradients \
    --clip_gradients_value 15 \
    --micro_batch_duration 20 \
    --micro_batch_number 3 \
    --max_allowed_utterance_gap 10.0 \
    --wandb_id '' \
    --save_top_k 1 \
    --schedular_data './checkpoints/lm_test.json' \
    --project_name 'deliberation-LM' \
    --label_smoothing 0.0 \
    



#GCC/9.3.0
#    --wandb_id '3bc1yvgb' \

#checkpoint_26_id_78.pt

# step_size = step size up, step size down is step_size*4 
