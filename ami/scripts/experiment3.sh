python train_sc_ctc.py \
    --checkpoint_dir './checkpoints' \
    --checkpoint '' \
    --model_config '../model_configs/experiment3_small.yaml' \
    --batch_length 30 \
    --min_lr 1e-5 \
    --max_lr 4e-4  \
    --step_size 250 \
    --accumulate_gradients 18 \
    --schedular_data 'experiment3.json' \
  


#
#python train_sc_ctc.py \
#    --checkpoint_dir '/fastdata/acp21rjf/ami_acc_sc_ctc_comp/' \
#    --checkpoint '' \
#    --model_config '../model_configs/conformer_compositional_sc_ctc_bpe_smallish.yaml' \
#    --batch_length 1950 \
#    --min_lr 1e-5 \
#    --max_lr 4e-4  \
#    --step_size 250 \
#    --accumulate_gradients 4 \
#    --schedular_data 'ctc_comp_smallish_scheduler.json' \

#python train_sc_ctc.py \
#    --checkpoint_dir '/fastdata/acp21rjf/ami_acc_sc_ctc_comp_discarding/' \
#    --checkpoint '' \
#    --model_config '../model_configs/conformer_compositional_sc_ctc_bpe_smallish_discard.yaml' \
#    --batch_length 1950 \
#    --min_lr 1e-5 \
#    --max_lr 4e-4  \
#    --step_size 250 \
#    --accumulate_gradients 4 \
#    --schedular_data 'ctc_comp_smallish_scheduler_discarding.json' \

