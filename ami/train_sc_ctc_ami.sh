echo '<<< TRAINING SMALL SC-CTC MODEL >>>'

python train_H.py --checkpoint '' \
    --checkpoint_dir './checkpoints' \
    --model_config '../model_configs/conformer_sc_ctc_bpe_small.yaml' \
    --min_lr 1e-5 \
    --max_lr 3e-4 \
    --step_size 150 \
    --accumulate_gradients 4 \
    --clip_gradients \
    --clip_gradients_value 10 \
    --micro_batch_duration 0 \
    --micro_batch_number 45 \
    --schedular_data 'sc-ctc_ami_baseline_scheduler.json' \
    --do_not_pass_segment_lens \
    --wandb_id '' 
    #--wandb_project 'PROJECT_NAME_GOES_HERW' \
    

echo '<<< WE ARE DONE! >>>'



# micro_batch_number = batch size 
# unless micro_batch_duration is > 0 then utterances from the same discourse are passed together up to a max duration
# step_size = step size up, step size down is step_size*4 
