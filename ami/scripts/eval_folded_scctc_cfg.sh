python eval_ctc_reuse.py  \
    --checkpoint_dir '../checkpoints_done/ami_acc_sc_ctc_folded' \
    --checkpoint 'checkpoint_326_id_5.pt' \
    --model_config '../model_configs/conformer_folded_sc_ctc_bpe_smallish.yaml' \
    --batch_size 4 \
    -lm 'ngrams/3gram-6mix.arpa' \
    --beam_size 150 \
    --beam_prune_logp -10.466836794439956 \
    --token_min_logp -4.178187657336318 \
    --split 'test' \
    --alpha 0.5 \
    --beta 0.8 \
    -sc 

    
    
    
#-sc 
#../checkpoints_done/single-domain/small_ctc 22.5
#--checkpoint_dir '../checkpoints_done/' ngrams/3gram-6mix.arpa
#3gram.kn012.arpa

#    --alpha 0.8205284495722655 \
#    --beta 0.806623980119611 \0.2387140687994136
