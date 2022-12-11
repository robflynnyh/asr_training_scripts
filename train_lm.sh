echo '<<< TRAINING AUTOREGRESSIVE TLM >>>'

python train_LM.py --checkpoint '' \
    --checkpoint_dir './checkpoints/abspostest/' \
    --model_config './experiment_configs/lm/decoder_test.yaml' \
    --min_lr 1e-5 \
    --max_lr 3.3e-4 \
    --step_size 760 \
    --accumulate_gradients 3\
    --clip_gradients \
    --clip_gradients_value 15 \
    --micro_batch_duration 30 \
    --micro_batch_number 10 \
    --max_allowed_utterance_gap 10.0 \
    --wandb_id '' \
    --save_top_k 1 \
    --schedular_data './checkpoints/tedlium_aptest.json' \
    --project_name 'deliberation-LM' \
    --label_smoothing 0.0 \
    



#GCC/9.3.0
#    --wandb_id '3bc1yvgb' \

#checkpoint_26_id_78.pt

# step_size = step size up, step size down is step_size*4 
