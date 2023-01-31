echo '<<< TRAINING AUTOREGRESSIVE TLM >>>'

python train_LM.py --checkpoint 'pg_19_pretrained_c_62_id_87.pt' \
    --checkpoint_dir './checkpoints/1kw_pg19checkpoints_nths/' \
    --model_config './experiment_configs/lm/decoder_pg19.yaml' \
    --min_lr 1e-6 \
    --max_lr 2.25e-4 \
    --step_size 700 \
    --accumulate_gradients 4 \
    --clip_gradients \
    --clip_gradients_value 15 \
    --micro_batch_duration 180 \
    --micro_batch_number 10 \
    --max_allowed_utterance_gap 10.0 \
    --wandb_id '' \
    --save_top_k 1 \
    --schedular_data './pg191kwftpd10.json' \
    -no_eos \
    --project_name 'FINETUNE-PG19-INTERSPEECH' \
    --label_smoothing 0.0 \
    



#GCC/9.3.0
#    --wandb_id '3bc1yvgb' \

#checkpoint_26_id_78.pt

# step_size = step size up, step size down is step_size*4 
