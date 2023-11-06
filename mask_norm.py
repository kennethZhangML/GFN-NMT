import torch 

def mask_and_normalize(self, s, probs, prev_tokens):
    """
    Modify the mask_and_normalize function to handle NMT-specific constraints.
    
    Args:
        s: An NxD matrix representing N states.
        
        probs: An NxA matrix of action probabilities.
        
        prev_tokens: A list of previously generated tokens in the target sequence.
    """
    oov_mask = torch.tensor([token not in self.env.target_vocab for token in prev_tokens], dtype = torch.float32).unsqueeze(1)
    
    probs = probs * (1 - oov_mask)
    
    probs = probs / probs.sum(1).unsqueeze(1)
    
    return probs
