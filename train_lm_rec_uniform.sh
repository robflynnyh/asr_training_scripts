echo '<<< TRAINING AUTOREGRESSIVE TLM >>>'

python rec_rescoring_uniformer.py --checkpoint '' \
    --train_hyp 'train_ted_lm.pkl' \
    --dev_hyp 'dev_ted_lm.pkl' \
    --checkpoint_dir './checkpoints/unitformer_test/' \
    --config './experiment_configs/lm/uniformer_test.yaml' \
    --min_lr 5e-5 \
    --max_lr 5e-5 \
    --step_size 750 \
    --clip_gradients \
    --clip_gradients_value 5 \
    --max_allowed_utterance_gap 10.0 \
    --wandb_id '' \
    --save_top_k 60 \
    --schedular_data './unitformer.json' \
    --utts 4 \
    -batch 40  \
    --optimizer_type 'ranger' \

    



#GCC/9.3.0
#    --wandb_id '3bc1yvgb' \

#checkpoint_26_id_78.pt

# step_size = step size up, step size down is step_size*4 
