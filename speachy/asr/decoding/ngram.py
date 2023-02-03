

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
        beam_prune_logp = -15, 
        token_min_logp = -8,
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
                'text': beam.text,
                'ngram_score': beam.lm_score - beam.logit_score,
                'am_score': beam.logit_score,
                'score': beam.lm_score # # score = ngram_score + am_score
            }
        decoded_data.append(decoded)

    return decoded_data