python eval.py \
    --checkpoint_dir ../checkpoints_done/single-domain/small_transducer_ctc_aux/ \
    --checkpoint 'checkpoint_129_id_8.pt' \
    --batch_size 1 \
    --split 'test' \
    -greedy
    
    
#-sc 
#../checkpoints_done/single-domain/small_ctc
#--checkpoint_dir '../checkpoints_done/'
#3gram.kn012.arpa

#    --alpha 0.8205284495722655 \
#    --beta 0.806623980119611 \
