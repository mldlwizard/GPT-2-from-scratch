import argparse
import yaml
from src.train import train
from src.inference import generate_text

def main():
    parser = argparse.ArgumentParser(description='GPT-2 Training and Inference')
    parser.add_argument('--config', type=str, default='config/default_config.yaml', help='Path to config file')
    parser.add_argument('--mode', type=str, choices=['train', 'inference'], required=True, help='Mode of operation')
    parser.add_argument('--prompt', type=str, help='Prompt for text generation (inference mode only)')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    if args.mode == 'train':
        train(config)
    elif args.mode == 'inference':
        if not args.prompt:
            raise ValueError("Prompt is required for inference mode")
        generate_text(config, args.prompt)

if __name__ == '__main__':
    main()