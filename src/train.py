import torch
import os
from tqdm import tqdm
from .model import GPT2, GPTConfig
from .data_loader import DataLoader
from .utils import setup_logging

def train(config):
    logger = setup_logging('train')
    device = torch.device(config['training']['device'])
    
    model = GPT2(GPTConfig(**config['model']))
    model.to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['training']['learning_rate'])
    data_loader = DataLoader(config)
    
    num_epochs = config['training']['num_epochs']
    log_interval = config['logging']['log_interval']
    save_interval = config['logging']['save_interval']
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for step in tqdm(range(data_loader.tokens.size(0) // (config['training']['batch_size'] * config['training']['sequence_length']))):
            x, y = data_loader.next_batch()
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            _, loss = model(x, y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if (step + 1) % log_interval == 0:
                avg_loss = total_loss / log_interval
                logger.info(f"Epoch {epoch+1}, Step {step+1}, Loss: {avg_loss:.4f}")
                total_loss = 0
            
            if (step + 1) % save_interval == 0:
                save_path = os.path.join(config['logging']['model_save_path'], f"model_epoch{epoch+1}_step{step+1}.pt")
                torch.save(model.state_dict(), save_path)
                logger.info(f"Model saved to {save_path}")

    logger.info("Training completed.")