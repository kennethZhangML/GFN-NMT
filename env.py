import torch
import torch.nn.functional as F

class NMTEncodingEnv:
    def __init__(self, src_vocab_size, tgt_vocab_size, max_seq_len):
        """
        Initializes an NMT environment for GFlowNet.

        Args:
            src_vocab_size: Size of the source language vocabulary.
            tgt_vocab_size: Size of the target language vocabulary.
            max_seq_len: Maximum sequence length for both source and target sequences.
        """
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        self.max_seq_len = max_seq_len
        self.num_actions = tgt_vocab_size  
    def mask(self, state):
        """
        Masks illegal actions (out-of-vocabulary tokens) in the state space.

        Args:
            state: A tensor representing the current state (source sequence).

        Returns:
            A binary mask tensor indicating valid actions for each state.
        """
        mask = (state >= 0) & (state < self.tgt_vocab_size)
        return mask

    def update(self, state, action):
        """
        Transition function that updates the state based on the selected action.

        Args:
            state: A tensor representing the current state (source sequence).
            action: An integer representing the selected action (target language token).

        Returns:
            A new state after applying the action.
        """
        new_state = torch.cat([state, action.unsqueeze(1)], dim = 1)
        
        new_state = new_state[:, :self.max_seq_len]

        return new_state

    def reward(self, state):
        """
        Computes the reward based on the translation quality (simplified for demonstration).

        Args:
            state: A tensor representing the final state (target sequence).

        Returns:
            A reward value.
        """
        reward = torch.randn(state.size(0))  
        return reward
