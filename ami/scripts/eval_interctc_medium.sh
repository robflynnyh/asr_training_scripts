python eval_ctc_reuse.py  \
    --checkpoint_dir '../checkpoints_done/single-domain/inter_ctc_nearly_medium' \
    --checkpoint 'inter_ctc_medium_81.pt' \
    --model_config '../model_configs/conformer_sc_ctc_bpe_nearly_medium.yaml' \
    --batch_size 3 \
    -lm '' \
    --beam_size 150 \
    --beam_prune_logp -10.466836794439956 \
    --token_min_logp -4.178187657336318 \
    --split 'test' \
    --alpha 0.45 \
    --beta 0.8 \
    -sc \
    --load_logits './eval_logits/test_interctc_medium_81.pkl' \
    

    
    
    
#-sc 
#../checkpoints_done/single-domain/small_ctc
#--checkpoint_dir '../checkpoints_done/'
#3gram.kn012.arpa

#    --alpha 0.8205284495722655 \
#    --beta 0.806623980119611 \0.2387140687994136
