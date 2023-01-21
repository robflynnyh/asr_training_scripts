

def decode_lm(logits_list, decoder, beam_width=100, encoded_lengths=None):
    decoded = []
    if encoded_lengths is None:
        encoded_lengths = [len(logits) for logits in logits_list]

    for logits, length in zip(logits_list, encoded_lengths):
        decoded.append(decoder.decode(logits[:length], beam_width=beam_width))
        
    return decoded


def decode_beams_lm(
        logits_list, 
        decoder, 
        beam_width=100, 
        encoded_lengths=None,
        beam_prune_logp = -10000, # no pruning
        token_min_logp = -5,
        prune_history = False,
    ):
    decoded_data = []
    if encoded_lengths is None:
        encoded_lengths = [len(logits) for logits in logits_list]

    for logits, length in zip(logits_list, encoded_lengths):
        beams = decoder.decode_beams(
            logits = logits[:length],
            beam_prune_logp = beam_prune_logp, 
            token_min_logp = token_min_logp,
            beam_width = beam_width,
            prune_history = prune_history
        )
        decoded = {}
        for i, beam in enumerate(beams):
            decoded[i] = {
                'text': beam[0],
                'ngram_score': beam[-1] - beam[-2], # score = ngram_score + am_score
                'am_score': beam[-2],
                'score': beam[-1]
            }
        decoded_data.append(decoded)

    return decoded_data