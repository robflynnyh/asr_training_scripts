var1="test"
var2="dev"
for context_size in 50 100 250 500
do
    for split in "${var1}" "${var2}"
    do
        echo "Running beam search for ${split} with context size ${context_size} (teacher forcing)"
        python beam_search_eval_crossutt.py \
            --checkpoint_dir ../checkpoints_done/ASR/TEDLIUM/checkpoints_cosine_noths/ \
            --checkpoint checkpoint_359_id_52.pt \
            --split ${split} \
            --teacher_forcing \
            -cache_len ${context_size} \
            > beamsearch_out/ted_${split}_beam_${context_size}_teacher_forcing.txt
    done
done

