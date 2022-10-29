python eval_ctc_reuse.py  \
    --checkpoint_dir '../checkpoints_done/hconf_BRN_mempos_memnotinconv/' \
    --checkpoint 'checkpoint_20_id_36.pt' \
    --model_config '' \
    --batch_size 12 \
    -lm '' \
    --beam_size 1 \
    --split 'test' \
    --alpha 0.75 \
    --beta 0.8 \
    --config_from_checkpoint_dir \
    --shuffle \
    -sc \
    --ctx_model \
  

    
    
#./ngrams/3gram-6mix.arpa
#-sc 
#../checkpoints_done/single-domain/small_ctc
#--checkpoint_dir '../checkpoints_done/'
