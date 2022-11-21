

def decode_lm(logits_list, decoder, beam_width=100, encoded_lengths=None):
    decoded = []
    if encoded_lengths is None:
        encoded_lengths = [len(logits) for logits in logits_list]

    for logits, length in zip(logits_list, encoded_lengths):
        decoded.append(decoder.decode(logits[:length], beam_width=beam_width))
        
    return decoded


def decode_beams_lm(logits_list, decoder, beam_width=100, encoded_lengths=None):
    decoded_text = []
    scores = []
    if encoded_lengths is None:
        encoded_lengths = [len(logits) for logits in logits_list]

    for logits, length in zip(logits_list, encoded_lengths):
        beams = decoder.decode_beams(
            logits = logits[:length],
            beam_width = beam_width,
        )
        text = [el[0] for el in beams]
        scores = [el[3] for el in beams]
        decoded_text.append(text)
        scores.append(scores)

    return decoded_text, scores