python eval_ctc_reuse.py  \
    --checkpoint_dir '../checkpoints_done/tosort/checkpoints_normal_talking_heads/' \
    --checkpoint 'checkpoint_44_id_8.pt' \
    --batch_size 5 \
    -lm '' \
    --beam_size 1 \
    --split 'test' \
    --alpha 0.75 \
    --beta 0.8 \
    --config_from_checkpoint_dir \
  

    
    
#./ngrams/3gram-6mix.arpa
#-sc 
#../checkpoints_done/single-domain/small_ctc
#--checkpoint_dir '../checkpoints_done/'
