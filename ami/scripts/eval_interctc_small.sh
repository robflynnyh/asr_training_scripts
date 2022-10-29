python eval_ctc_reuse.py  \
    --checkpoint_dir '../checkpoints_done/single-domain/inter_ctc_small' \
    --checkpoint 'inter_checkpoint_251_id_45.pt' \
    --model_config '../model_configs/conformer_inter_sc_ctc_bpe_small.yaml' \
    --batch_size 4 \
    -lm '' \
    --beam_size 150 \
    --beam_prune_logp -10.466836794439956 \
    --token_min_logp -4.178187657336318 \
    --split 'test' \
    --alpha 0.5 \
    --beta 0.8 \
    -sc \
    --load_logits './eval_logits/test_inter_ctc_small_251.pkl' \
    

    
#-sc 
#../checkpoints_done/single-domain/small_ctc
#--checkpoint_dir '../checkpoints_done/' ngrams/3gram-6mix.arpa
#3gram.kn012.arpa

#    --alpha 0.8205284495722655 \
#    --beta 0.806623980119611 \0.2387140687994136
"""
    python eval_ctc_reuse.py  \
        --checkpoint_dir '../checkpoints_done/single-domain/ctc_medium' \
        --checkpoint 'checkpoint_170_id_15.pt' \
        --model_config '../model_configs/conformer_ctc_bpe_medium.yaml' \
        --batch_size 3 \
        -lm '' \
        --beam_size 150 \
        --beam_prune_logp -10.466836794439956 \
        --token_min_logp -4.178187657336318 \
        --split 'test' \
        --alpha 0.6 \
        --beta 0.8     


python eval_ctc_reuse.py  \
    --checkpoint_dir '../checkpoints_done/single-domain/ami_sc_ctc_large' \
    --checkpoint 'checkpoint_85_id_15.pt' \
    --model_config '../model_configs/conformer_sc_ctc_bpe_large.yaml' \
    --batch_size 1 \
    -lm '' \
    --beam_size 150 \
    --beam_prune_logp -10.466836794439956 \
    --token_min_logp -4.178187657336318 \
    --split 'test' \
    --alpha 0.6 \
    --beta 0.8  \
    -sc 
    """