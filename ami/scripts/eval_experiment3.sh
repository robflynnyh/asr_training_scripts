python eval_ctc_reuse.py  \
    --checkpoint_dir '../checkpoints_done/experiment3_small' \
    --checkpoint 'checkpoint_170_id_75.pt' \
    --model_config '../model_configs/experiment3_sc_small.yaml' \
    --batch_size 10 \
    -lm '' \
    --beam_size 2 \
    --split 'dev' \
    --alpha 0.65 \
    --beta 0.8 \
    -sc \
    --load_logits './eval_logits/dev_exp3_ctc_170.pkl'



    
    
    
#-sc 
#../checkpoints_done/single-domain/small_ctc
#--checkpoint_dir '../checkpoints_done/'
#3gram.kn012.arpa

#    --alpha 0.8205284495722655 \
#    --beta 0.806623980119611 \
