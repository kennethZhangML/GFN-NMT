import torch
import random
from metrics import *
from env import NMTEncodingEnv
from Transformer import *

def test_nmt_environment_and_model(encoder, decoder, env, src_seq, max_tgt_seq_len=20):
    """
    Test function to generate translations and calculate BLEU scores using the NMT environment and model.

    Args:
        encoder: The NMT encoder model.
        decoder: The NMT decoder model.
        env: The NMT environment.
        src_seq: The source sequence for translation.
        max_tgt_seq_len: The maximum length of the target sequence to generate.

    Returns:
        bleu_score: The BLEU score for the generated translation.
    """
    encoder.eval()
    decoder.eval()

    tgt_seq = [env.tgt_vocab_size - 2]  

    src_seq = torch.tensor(src_seq).unsqueeze(0)  
    hidden, cell = encoder(src_seq)

    for _ in range(max_tgt_seq_len):
        input = torch.tensor(tgt_seq[-1]).unsqueeze(0)  
        output, hidden, cell = decoder(input, hidden, cell)
        top1 = output.argmax(1).item()  
        tgt_seq.append(top1)

        if top1 == env.tgt_vocab_size - 1:
            break

    reference = tgt_seq[1:-1]  
    bleu_score = calculate_bleu(reference, tgt_seq[1:-1])  

    return bleu_score

# TODO: Finish Testing Client
encoder = Encoder(...)
decoder = Decoder(...)
nmt_env = NMTEncodingEnv(15000, 15000, 32)

src_sequence = [random.randint(0, NMTEncodingEnv.src_vocab_size - 1) for _ in range(10)]

bleu_score = test_nmt_environment_and_model(encoder, decoder, nmt_env, src_sequence)
print(f"BLEU Score: {bleu_score}")
