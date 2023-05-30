echo '<<< TRAINING AUTOREGRESSIVE FEEDBACK TLM >>>'

python train_lm_pg19.py --checkpoint '' \
    --checkpoint_dir './checkpoints/' \
    --config './experiment_configs/pg19_s4ormer.yaml' \
    --min_lr 1e-4 \
    --max_lr 1e-4 \
    --step_size 250 \
    --accumulate_gradients 1 \
    --clip_gradients \
    --clip_gradients_value 0.5 \
    --batch 3 \
    --wandb_id '' \
    --save_top_k 5 \
    --schedular_data './pg191kwftpd90.json' \
    --project_name 'FINETUNE-PG19-INTERSPEECH' \
    --mixed_precision \
    



#GCC/9.3.0
#    --wandb_id '3bc1yvgb' \

#checkpoint_26_id_78.pt

# step_size = step size up, step size down is step_size*4 
