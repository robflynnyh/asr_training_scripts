python eval_ctc.py  \
    --checkpoint_dir '../checkpoints_done/ctc_small_1000/' \
    --checkpoint 'checkpoint_196_id_58.pt' \
    --model_config '../model_configs/conformer_ctc_1000_small.yaml' \
    --batch_size 5 \
    -lm './ngrams/3gram-6mix.arpa' \
    --beam_size 100 \
    --split 'test' \
    --alpha 0.7 \
    --beta 1.0 \



python eval_ctc.py  \
    --checkpoint_dir './checkpoints/' \
    --checkpoint 'checkpoint_8_id_63.pt' \
    --model_config '../model_configs/Hconformer_ctc_bpe_small.yaml' \
    --batch_size 5 \
    -lm '' \
    --beam_size 100 \
    --split 'test' \
    --alpha 0.7 \
    --beta 1.0 \

  

    
    
    
#-sc 
#../checkpoints_done/single-domain/small_ctc
#--checkpoint_dir '../checkpoints_done/'
