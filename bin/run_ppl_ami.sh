python -m speachy.lm.scripts.eval_perplexity --checkpoint ./checkpoints/open_sub_ft_ami/checkpoint_809_id_37.pt   --config ./experiment_configs/lm/decoder_pg19_sep_token_ami.yaml --max_cache 0 --split 'test' > ./ppls/ppl_0_test.txt
python -m speachy.lm.scripts.eval_perplexity --checkpoint ./checkpoints/open_sub_ft_ami/checkpoint_809_id_37.pt   --config ./experiment_configs/lm/decoder_pg19_sep_token_ami.yaml --max_cache 0 --split 'dev' > ./ppls/ppl_0_dev.txt

python -m speachy.lm.scripts.eval_perplexity --checkpoint ./checkpoints/open_sub_ft_ami/checkpoint_809_id_37.pt   --config ./experiment_configs/lm/decoder_pg19_sep_token_ami.yaml --max_cache 50 --split 'test' > ./ppls/ppl_50_test.txt
python -m speachy.lm.scripts.eval_perplexity --checkpoint ./checkpoints/open_sub_ft_ami/checkpoint_809_id_37.pt   --config ./experiment_configs/lm/decoder_pg19_sep_token_ami.yaml --max_cache 50 --split 'dev' > ./ppls/ppl_50_dev.txt

python -m speachy.lm.scripts.eval_perplexity --checkpoint ./checkpoints/open_sub_ft_ami/checkpoint_809_id_37.pt   --config ./experiment_configs/lm/decoder_pg19_sep_token_ami.yaml --max_cache 100 --split 'test' > ./ppls/ppl_100_test.txt
python -m speachy.lm.scripts.eval_perplexity --checkpoint ./checkpoints/open_sub_ft_ami/checkpoint_809_id_37.pt   --config ./experiment_configs/lm/decoder_pg19_sep_token_ami.yaml --max_cache 100 --split 'dev' > ./ppls/ppl_100_dev.txt

python -m speachy.lm.scripts.eval_perplexity --checkpoint ./checkpoints/open_sub_ft_ami/checkpoint_809_id_37.pt   --config ./experiment_configs/lm/decoder_pg19_sep_token_ami.yaml --max_cache 250 --split 'test' > ./ppls/ppl_250_test.txt
python -m speachy.lm.scripts.eval_perplexity --checkpoint ./checkpoints/open_sub_ft_ami/checkpoint_809_id_37.pt   --config ./experiment_configs/lm/decoder_pg19_sep_token_ami.yaml --max_cache 250 --split 'dev' > ./ppls/ppl_250_dev.txt

python -m speachy.lm.scripts.eval_perplexity --checkpoint ./checkpoints/open_sub_ft_ami/checkpoint_809_id_37.pt   --config ./experiment_configs/lm/decoder_pg19_sep_token_ami.yaml --max_cache 500 --split 'test' > ./ppls/ppl_500_test.txt
python -m speachy.lm.scripts.eval_perplexity --checkpoint ./checkpoints/open_sub_ft_ami/checkpoint_809_id_37.pt   --config ./experiment_configs/lm/decoder_pg19_sep_token_ami.yaml --max_cache 500 --split 'dev' > ./ppls/ppl_500_dev.txt