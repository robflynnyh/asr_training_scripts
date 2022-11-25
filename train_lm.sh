echo '<<< TRAINING AUTOREGRESSIVE TLM >>>'

python train_LM.py --checkpoint 'pg_19_checkpoint_42_id_36.pt' \
    --checkpoint_dir './checkpoints/pg19checkpoints_dropout10_nths/' \
    --model_config './experiment_configs/lm/decoder_pg19.yaml' \
    --min_lr 1e-5 \
    --max_lr 3.3e-4 \
    --step_size 760 \
    --accumulate_gradients 4\
    --clip_gradients \
    --clip_gradients_value 15 \
    --micro_batch_duration 180 \
    --micro_batch_number 10 \
    --max_allowed_utterance_gap 10.0 \
    --wandb_id '' \
    --save_top_k 1 \
    --schedular_data './checkpoints/tedlium_pg19nths_ls.json' \
    --project_name 'deliberation-LM' \
    --label_smoothing 0.1 \
    



#GCC/9.3.0
#    --wandb_id '3bc1yvgb' \

#checkpoint_26_id_78.pt

# step_size = step size up, step size down is step_size*4 
