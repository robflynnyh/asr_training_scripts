echo '<<< TRAINING AUTOREGRESSIVE TLM >>>'

python train_LM.py --checkpoint '' \
    --checkpoint_dir './checkpoint_seqmasking_attention_40' \
    --model_config './lm/decoder_test.yaml' \
    --min_lr 1e-6 \
    --max_lr 2.75e-4 \
    --step_size 60 \
    --accumulate_gradients 3 \
    --clip_gradients \
    --clip_gradients_value 10 \
    --micro_batch_duration 60 \
    --micro_batch_number 10 \
    --max_allowed_utterance_gap 3.0 \
    --wandb_id '' \
    --save_top_k 1 \
    --schedular_data 'seqdrop_seqmasking_attention_40.json' 
    



#GCC/9.3.0
#    --wandb_id '3bc1yvgb' \

#checkpoint_26_id_78.pt

# step_size = step size up, step size down is step_size*4 
