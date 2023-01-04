

def get_max_length(dataloader):
    max_len = 0
    for batch in dataloader:
        max_len = max(max_len, batch['tokens'].shape[1])
    return max_len