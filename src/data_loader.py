import torch
import tiktoken

class DataLoader:
    def __init__(self, config):
        self.batch_size = config['training']['batch_size']
        self.sequence_length = config['training']['sequence_length']
        
        with open(config['data']['input_file'], 'r') as f:
            text = f.read()
        enc = tiktoken.get_encoding('gpt2')
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens)
        print(f"1 epoch = {len(self.tokens) // (self.batch_size * self.sequence_length)} batches")
        
        self.current_position = 0
    
    def next_batch(self):
        B, T = self.batch_size, self.sequence_length
        buf = self.tokens[self.current_position: self.current_position + B*T + 1]
        x = (buf[:-1]).view(B, T)
        y = (buf[1:]).view(B, T)

        self.current_position += B*T
        if self.current_position + (B*T+1) > len(self.tokens):
            self.current_position = 0
        return x, y