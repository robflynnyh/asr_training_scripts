python eval_ctc.py  \
    --checkpoint_dir '../checkpoints_done/single-domain/small_ctc/' \
    --checkpoint 'small_ctc_134.pt' \
    --model_config '../model_configs/conformer_ctc_bpe_small.yaml' \
    --batch_size 1 \
    -lm '' \
    --beam_size 100  \
    --split 'test' \
    --alpha 0.8 \
    --beta 0.8 \
    --tokenizer 'tokenizer_spe_char'
 
    
    
    
#-sc 
#../checkpoints_done/single-domain/small_ctc
#--checkpoint_dir '../checkpoints_done/'
#3gram.kn012.arpa

#    --alpha 0.8205284495722655 \
#    --beta 0.806623980119611 \
