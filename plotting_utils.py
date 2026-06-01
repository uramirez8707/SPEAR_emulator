import pandas as pd
import matplotlib.pyplot as plt

def plot_var_loss(train_df, val_df, output, fig_dir, ymin, ymax):

    plt.figure(figsize=(10, 6))
    label = f"_{output}"
    if output == "average":
        label = ""

    plt.plot(train_df['step'], train_df[f'train_loss{label}'], label='Training Loss', alpha=0.7)
    plt.plot(val_df['step'], val_df[f'val_loss{label}'], label='Validation Loss', marker='o', linewidth=2)

    plt.xlabel('Training Steps')
    plt.ylabel(f'MSE Loss - {output}')
    plt.title('SPEAR Emulator Training Curve')
    plt.ylim((ymin, ymax))

    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.savefig(f"{fig_dir}/losses.{output}.png", dpi=300, bbox_inches='tight')
    plt.close()

def get_ylim(df, output_channels):
    all_losses = []
    for output in output_channels:
        all_losses.extend(
            df[f'train_loss_{output}'].dropna().values
        )
        all_losses.extend(
            df[f'val_loss_{output}'].dropna().values
        )
    all_losses.extend(df['train_loss'].dropna().values)
    all_losses.extend(df['val_loss'].dropna().values)

    ymin = min(all_losses)
    ymax = max(all_losses)

    return ymin, ymax

def plot_loss(csv_path, fig_dir, label, output_channels):
    df = pd.read_csv(csv_path)
    columns = df.columns.tolist()

    ymin, ymax = get_ylim(df, output_channels)
    for output in output_channels:
        train_df = df[['step', f'train_loss_{output}']].dropna()
        val_df = df[['step', f'val_loss_{output}']].dropna()

        plot_var_loss(train_df, val_df, output, fig_dir, ymin, ymax)

    # Plot the average loss per step
    train_df = df[['step', 'train_loss']].dropna()
    val_df = df[['step', 'val_loss']].dropna()
    plot_var_loss(train_df, val_df, "average", fig_dir, ymin, ymax)
