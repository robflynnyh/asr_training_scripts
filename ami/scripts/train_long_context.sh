echo '<<< TRAINING SMALL CROSS-CONTEXT CTC MODEL >>>'

python train_H.py --checkpoint '' \
    --checkpoint_dir './checkpoints' \
    --model_config '/exp/exp1/acp21rjf/deliberation/Custom/model_configs/conformer_sc_ctc_bpe_small.yaml' \
    --min_lr 1e-6 \
    --max_lr 4e-5 \
    --step_size 1600 \
    --accumulate_gradients 8 \
    --clip_gradients \
    --clip_gradients_value 10 \
    --micro_batch_duration 35 \
    --micro_batch_number 6 \
    --schedular_data 'longcontext.json' \
    --do_not_pass_segment_lens \
    --concat_samples \
    --gap 0.1 \
    --speaker_gap 1.0 \
    --single_speaker_with_gaps \
    --split_speakers
    




#    --wandb_id '3bc1yvgb' \

#checkpoint_26_id_78.pt

# step_size = step size up, step size down is step_size*4 
