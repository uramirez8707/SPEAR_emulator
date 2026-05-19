import pandas as pd
import matplotlib.pyplot as plt

def plot_loss(csv_path, out_path):
    df = pd.read_csv(csv_path)
    train_df = df[['step', 'train_loss']].dropna()
    val_df = df[['step', 'val_loss']].dropna()

    plt.figure(figsize=(10, 6))
    plt.plot(train_df['step'], train_df['train_loss'], label='Training Loss', alpha=0.7)
    plt.plot(val_df['step'], val_df['val_loss'], label='Validation Loss', marker='o', linewidth=2)
    
    plt.xlabel('Training Steps')
    plt.ylabel('MSE Loss')
    plt.title('SPEAR Emulator Training Curve')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
