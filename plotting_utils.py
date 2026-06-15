import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_var_loss(train_df, val_df, output, fig_dir, ymin, ymax):

    plt.figure(figsize=(10, 6))
    label = f"_{output}"
    if output == "average":
        label = ""

    plt.plot(train_df['step'], train_df[f'train_loss{label}'], label='Training Loss', alpha=0.7)
    plt.plot(val_df['step'], val_df[f'val_loss{label}'], label='Validation Loss', marker='o', linewidth=2)

    plt.xlabel('Training Steps')
    plt.ylabel(f'MSE Loss - {output} [normalized units]')
    plt.title('Training and Validation losses over training steps')
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

def get_variable_groups(output_channels):
    temperatures = []
    u_components = []
    v_components = []
    specific_humidity = []
    others = []
    for output in output_channels:
        if "air_temperature" in output:
            temperatures.append(output)
        elif "northward_wind" in output:
            v_components.append(output)
        elif "eastward_wind" in output:
            u_components.append(output)
        elif "specific_humidity" in output:
            specific_humidity.append(output)
        else:
            others.append(output)

    groups = [temperatures, u_components, v_components, specific_humidity, others]
    group_labels = ["air_temperature", "northward_wind", "eastward_wind", "specific_humidity", "surface"]

    return group_labels, groups

def plot_loss_heat_map(csv_path, fig_dir, label, output_channels):
    df = pd.read_csv(csv_path)

    group_labels, groups = get_variable_groups(output_channels)

    ordered_outputs = []
    for group in groups:
        ordered_outputs.extend(group)

    val_cols = [f"val_loss_{out}" for out in ordered_outputs if f"val_loss_{out}" in df.columns]
    val_df = df.dropna(subset=val_cols, how="all")
    eval_steps = np.sort(val_df["step"].dropna().unique())

    loss_matrix = []
    row_labels = []

    for output in ordered_outputs:
        col = f"val_loss_{output}"

        if col not in df.columns:
            print(f"Missing {col}")
            continue

        tmp = (
            df[["step", col]]
            .dropna()
            .groupby("step")[col]
            .mean()
            .reindex(eval_steps)
            )

        loss_matrix.append(tmp.values.astype(float))
        row_labels.append(output)

    loss_matrix = np.array(loss_matrix)
    fig, ax = plt.subplots(
        figsize=(14, max(8, 0.35 * len(row_labels)))
    )

    sns.heatmap(
        loss_matrix,
        ax=ax,
        cmap="viridis",
        xticklabels=False,
        yticklabels=row_labels,
        cbar_kws={"label": "Validation loss"},
    )

    # group separators (fix indexing safety)
    current = 0
    for group in groups[:-1]:
        current += len(group)
        ax.axhline(current, color="white", lw=1.5)

    ax.set_xlabel("Training step")
    ax.set_ylabel("Output variable")
    ax.set_title(f"{label}: Validation Loss per Variable")

    plt.tight_layout()

    os.makedirs(fig_dir, exist_ok=True)

    fig.savefig(
        os.path.join(fig_dir, f"{label}_loss_heatmap.png"),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)

