echo '<<< TRAINING AUTOREGRESSIVE TLM >>>'

python train_LM.py --checkpoint 'checkpoint_34_id_27.pt' \
    --checkpoint_dir './checkpoints/pg19_d010_nths/' \
    --model_config './experiment_configs/lm/decoder_pg19.yaml' \
    --min_lr 1e-5 \
    --max_lr 3.3e-4 \
    --step_size 400 \
    --accumulate_gradients 10\
    --clip_gradients \
    --clip_gradients_value 15 \
    --micro_batch_duration 300 \
    --micro_batch_number 3 \
    --max_allowed_utterance_gap 10.0 \
    --wandb_id '' \
    --save_top_k 1 \
    --schedular_data './checkpoints/tedlium_pg19nths.json' \
    --project_name 'deliberation-LM' \
    



#GCC/9.3.0
#    --wandb_id '3bc1yvgb' \

#checkpoint_26_id_78.pt

# step_size = step size up, step size down is step_size*4 
