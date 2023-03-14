for context_size in 0 50 100 250 500
do

    python -m speachy.rescoring.scripts.rescore_with_TLM_v5_batched \
        --checkpoint ./checkpoints/open_sub_ft_ami/checkpoint_809_id_37.pt \
        -hyp_dev ./ami/nbests/n_best_dev_ami.pkl \
        -hyp_test ./ami/nbests/AMI_TEST_NBEST.pkl \
        -batch_size 50 \
        --config ./experiment_configs/lm/decoder_pg19_sep_token_ami.yaml \
        -history ${context_size} \
        --stop_at_beam 100  \
        --half_precision \
        -bpe_lm_weight 0.4 \
        -bpe_len_pen 2.4 \
        -ngram_scale 0.9 \
        -tlm_scale 11.4 \
        -tlm_mean -43.6 \
        -tlm_std 49.8 \
        > ./rescore/rescore_out_ami_$context_size.txt

    python -m speachy.rescoring.scripts.rescore_with_TLM_v5_batched \
        --checkpoint ./checkpoints/open_sub_ft_ami/checkpoint_809_id_37.pt \
        -hyp_dev ./ami/nbests/n_best_dev_ami.pkl \
        -hyp_test ./ami/nbests/AMI_TEST_NBEST.pkl \
        -batch_size 50 \
        --config ./experiment_configs/lm/decoder_pg19_sep_token_ami.yaml \
        -history ${context_size} \
        --stop_at_beam 100  \
        --half_precision \
        -bpe_lm_weight 0.4 \
        -bpe_len_pen 2.4 \
        -ngram_scale 0.9 \
        -tlm_scale 11.4 \
        -tlm_mean -43.6 \
        -tlm_std 49.8 \
        -use_targets \
        > ./rescore/rescore_out_TARGETS_ami_$context_size.txt
done



'''
    python -m speachy.rescoring.scripts.rescore_with_TLM_v5_batched \
        --checkpoint ./checkpoints/open_sub_ft_ted/ft_ted_checkpoint_1259_id_36.pt \
        -hyp_dev ./tedlium/n_bests/dev_tmp_out.pkl \
        -hyp_test ./tedlium/n_bests/test_tmp.pkl \
        -batch_size 100 \
        --config ./experiment_configs/lm/decoder_pg19_sep_token_ted_am.yaml \
        -history ${context_size} \
        --stop_at_beam 100  \
        --half_precision \
        -bpe_lm_weight -0.4 \
        -bpe_len_pen 2.3 \
        -ngram_scale 1.4 \
        -tlm_scale 35.1 \
        -tlm_mean -175.6 \
        -tlm_std 106.8 \
        > ./rescore/rescore_out_ted_$context_size.txt

    python -m speachy.rescoring.scripts.rescore_with_TLM_v5_batched \
        --checkpoint ./checkpoints/open_sub_ft_ted/ft_ted_checkpoint_1259_id_36.pt \
        -hyp_dev ./tedlium/n_bests/dev_tmp_out.pkl \
        -hyp_test ./tedlium/n_bests/test_tmp.pkl \
        -batch_size 100 \
        --config ./experiment_configs/lm/decoder_pg19_sep_token_ted_am.yaml \
        -history ${context_size} \
        --stop_at_beam 100  \
        --half_precision \
        -bpe_lm_weight -0.4 \
        -bpe_len_pen 2.3 \
        -ngram_scale 1.4 \
        -tlm_scale 35.1 \
        -tlm_mean -175.6 \
        -tlm_std 106.8 \
        -use_targets \
        > ./rescore/rescore_out_TARGETS_ted_$context_size.txt

'''