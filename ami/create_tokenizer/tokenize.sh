python process_asr_text_tokenizer.py --manifest="train.manifest" \
         --data_root="./" \
         --vocab_size=100000 \
         --spe_type='word' \
         --spe_bos \
         --spe_character_coverage 0.98 \
         --tokenizer="spe" \
         