python eval_ctc.py  \
    --checkpoint_dir './checkpoints' \
    --checkpoint 'checkpoint_94_id_15.pt' \
    --model_config '../model_configs/conformer_compositional_sc_ctc_bpe_smallish_discard.yaml' \
    --batch_size 1   \
    -lm '' \
    --beam_size 10 \
    --split 'test' \
    --alpha 0.6 \
    --beta 0.8 \
    -sc

    
    
    
#-sc 
#../checkpoints_done/single-domain/small_ctc
#--checkpoint_dir '../checkpoints_done/'
#3gram.kn012.arpa

#    --alpha 0.8205284495722655 \
#    --beta 0.806623980119611 \