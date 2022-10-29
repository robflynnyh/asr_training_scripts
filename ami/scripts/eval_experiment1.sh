python eval_ctc.py  \
    --checkpoint_dir '../checkpoints_done/experiment_1.1' \
    --checkpoint 'checkpoint_105_id_69.pt' \
    --model_config '../model_configs/conformer_compositional_sc_ctc_bpe_smallish.yaml' \
    --batch_size 5 \
    -lm '' \
    --beam_size 100 \
    --split 'dev' \
    --alpha 0.4 \
    --beta 0.8 \
    -sc

    
    
    
#-sc 
#../checkpoints_done/single-domain/small_ctc
#--checkpoint_dir '../checkpoints_done/'
#3gram.kn012.arpa

#    --alpha 0.8205284495722655 \
#    --beta 0.806623980119611 \
