python eval_ctc.py  \
    --checkpoint_dir '../checkpoints_done/experiment_2.1' \
    --checkpoint 'checkpoint_156_id_93.pt' \
    --model_config '../model_configs/experiment2_sc_small.yaml' \
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
