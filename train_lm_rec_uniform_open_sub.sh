echo '<<< TRAINING AUTOREGRESSIVE TLM >>>'

python rec_rescoring_opensub_uniformer.py --checkpoint '' \
    --checkpoint_dir './checkpoints/unitformer_test/' \
    --config './experiment_configs/lm/uniformer_test.yaml' \
    --min_lr 5e-4 \
    --max_lr 5e-4 \
    --step_size 1000 \
    --clip_gradients \
    --clip_gradients_value 0.5 \
    --max_allowed_utterance_gap 10.0 \
    --wandb_id '' \
    --save_top_k 100 \
    --schedular_data './unitformer.json' \
    --utts 1 \
    -batch 500  \

    



#GCC/9.3.0
#    --wandb_id '3bc1yvgb' \

#checkpoint_26_id_78.pt

# step_size = step size up, step size down is step_size*4 
