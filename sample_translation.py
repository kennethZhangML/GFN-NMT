import torch 
from torch.distributions import Categorical

def sample_translation(self, s0, max_length=50):
    """
    Perform translation using the forward_policy to generate the target sequence.
    
    Args:
        s0: An NxD matrix of initial states.
        
        max_length: Maximum length of the target sequence to generate (prevent infinite loops).
    
    Returns:
        translations: A list of translated target sequences.
    """
    s = s0.clone()
    translations = []
    
    for _ in range(max_length):
        probs = self.forward_probs(s)
        sampled_tokens = Categorical(probs).sample()
        
        translations.append(sampled_tokens.tolist())
        
        s = self.env.update_translation(s, sampled_tokens)
    return translations
