python eval_ctc_reuse.py  \
    --checkpoint_dir './checkpoints' \
    --checkpoint 'checkpoint_612_id_92.pt' \
    --model_config '../model_configs/conformer_ctc_char_small.yaml' \
    --batch_size 4 \
    -lm '' \
    --beam_size 1 \
    --split 'test' \
    --alpha 0.75 \
    --beta 0.8 \
  

    
    
#./ngrams/3gram-6mix.arpa
#-sc 
#../checkpoints_done/single-domain/small_ctc
#--checkpoint_dir '../checkpoints_done/'
