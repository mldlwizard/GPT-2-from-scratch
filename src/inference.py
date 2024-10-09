import torch
import tiktoken
from .model import GPT2, GPTConfig
from .utils import setup_logging

def generate_text(config, prompt, max_length=100):
    logger = setup_logging('inference')
    device = torch.device(config['training']['device'])
    
    model = GPT2(GPTConfig(**config['model']))
    model.to(device)
    model.eval()
    
    enc = tiktoken.get_encoding('gpt2')
    input_ids = torch.tensor(enc.encode(prompt)).unsqueeze(0).to(device)
    
    with torch.no_grad():
        for _ in range(max_length):
            outputs = model(input_ids)
            next_token_logits = outputs[0][:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            
            if next_token.item() == enc.encode('\n')[0]:
                break
    
    generated_text = enc.decode(input_ids[0].tolist())
    logger.info(f"Generated text: {generated_text}")
    return generated_text