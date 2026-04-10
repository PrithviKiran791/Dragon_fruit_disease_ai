import os
import torch
import sys

sys.path.insert(0, r'w:\latest mini project\Dragon_fruit_disease_ai')
from models.convitx import build_convitx_base

def create_model(model_name, num_classes):
    path = os.path.join(r'w:\latest mini project\Dragon_fruit_disease_ai\models', model_name)
    if not os.path.exists(path):
        print(f'Creating dummy {model_name}...')
        model = build_convitx_base(num_classes=num_classes, enforce_budget=False)
        torch.save(model.state_dict(), path)
        print(f'Saved {path}')
    else:
        print(f'{model_name} already exists.')

if __name__ == '__main__':
    create_model('best_convitx.pth', 6)
    create_model('quality_convitx.pth', 4)
