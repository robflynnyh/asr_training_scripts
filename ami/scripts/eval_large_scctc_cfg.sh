python eval_ctc.py  \
    --checkpoint_dir '../checkpoints_done/single-domain/ctc_medium' \
    --checkpoint 'checkpoint_170_id_15.pt' \
    --model_config '../model_configs/conformer_ctc_bpe_medium.yaml' \
    --batch_size 25 \
    -lm '' \
    --beam_size 500 \
    --split 'dev' \
    --alpha 0.6 \
    --beta 0.8 \

    
    
    
#-sc 
#../checkpoints_done/single-domain/small_ctc
#--checkpoint_dir '../checkpoints_done/'
#3gram.kn012.arpa

#    --alpha 0.8205284495722655 \
#    --beta 0.806623980119611 \