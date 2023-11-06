import nltk
from nltk.translate.bleu_score import sentence_bleu

def calculate_bleu(reference, translation):
    """
    Calculate BLEU score between a reference and a translation.
    
    Args:
        reference: A list of reference tokens (ground truth).
        translation: A list of generated tokens.
    
    Returns:
        bleu_score: The BLEU score between the reference and translation.
    """
    reference = [reference]
    translation = [translation]
    
    bleu_score = sentence_bleu(reference, translation)
    
    return bleu_score

def reward_function(reference, translation):
    """
    Reward function for NMT based on BLEU score.
    
    Args:
        reference: A list of reference tokens (ground truth).
        translation: A list of generated tokens.
    
    Returns:
        reward: The reward based on the BLEU score.
    """
    bleu_score = calculate_bleu(reference, translation)
    reward = bleu_score
    
    return reward
