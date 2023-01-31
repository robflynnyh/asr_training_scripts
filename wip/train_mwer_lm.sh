echo '<<< TRAINING AUTOREGRESSIVE TLM >>>'

python mwer_rescoring.py --checkpoint 'pg_19_pretrained_c_62_id_87.pt' \
    --train_hyp './train_no_noise_10.pkl' \
    --dev_hyp './hyps/dev_10k_lower_pruning.pkl' \
    --checkpoint_dir './checkpoints/1kw_pg19checkpoints_nths/' \
    --config './experiment_configs/lm/decoder_pg19.yaml' \
    --min_lr 1e-5 \
    --max_lr 4.0e-4 \
    --step_size 900 \
    --clip_gradients \
    --clip_gradients_value 15 \
    --batch_size 5 \
    --utts_per_sample 25 \
    --negatives 4 \
    --max_allowed_utterance_gap 10.0 \
    --wandb_id '' \
    --save_top_k 1 \
    --schedular_data './pg19mwerrr.json' \
    --project_name 'FINETUNE-PG19-INTERSPEECH' \

    



#GCC/9.3.0
#    --wandb_id '3bc1yvgb' \

#checkpoint_26_id_78.pt

# step_size = step size up, step size down is step_size*4 
