import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

def parse_pytorch_log(log_path):
    val_accuracies = []
    with open(log_path, 'r') as f:
        for line in f:
            if "Accuracy of the network on the" in line and "test images" in line:
                 match = re.search(r'([\d\.]+)%', line)
                 if match:
                    val_accuracies.append(float(match.group(1)))
    return val_accuracies

def parse_jittor_log(log_path):
    val_accuracies = []
    with open(log_path, 'r') as f:
        for line in f:
            if "Accuracy of the network on the" in line and "test images" in line:
                match = re.search(r'([\d\.]+)%', line)
                if match:
                    val_accuracies.append(float(match.group(1)))
    return val_accuracies

def plot_alignment_curves(pytorch_log, jittor_log, save_path='alignment_curves.png'):
    
    pt_acc = parse_pytorch_log(pytorch_log)
    jt_acc = parse_jittor_log(jittor_log)

    epochs = range(min(len(pt_acc), len(jt_acc)))
    min_len = len(epochs)
    pt_acc = pt_acc[:min_len]
    jt_acc = jt_acc[:min_len]
    
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(epochs, pt_acc, 'o-', label=f'PyTorch Baseline (Max: {max(pt_acc):.2f}%)', color='orangered')
    ax.plot(epochs, jt_acc, 's--', label=f'Jittor Implementation (Max: {max(jt_acc):.2f}%)', color='dodgerblue')
    
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_title('Jittor vs. PyTorch: Validation Accuracy Alignment', fontsize=16)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Validation Accuracy (%)', fontsize=12)
    ax.legend(fontsize=12)
    ax.grid(True)
    
    plt.savefig(save_path, dpi=300)

def parse_loss_time(log_path):

    train_losses = []
    epoch_times = []
    
    with open(log_path, 'r') as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            if "EPOCH" in line and "training takes" in line and i > 0:
                time_match = re.search(r'takes\s+([\d|:]+)', line)
                if time_match:
                    try:
                        h, m, s = map(int, time_match.group(1).split(':'))
                        total_seconds = h * 3600 + m * 60 + s
                        epoch_times.append(total_seconds)
                    except ValueError:
                        epoch_times.append(0)

                prev_line = lines[i-1]
                loss_match = re.search(r'loss\s+[\d\.]+\s+\(([\d\.]+)\)', prev_line)
                if loss_match:
                    train_losses.append(float(loss_match.group(1)))
                else:
                    train_losses.append(0)
    return train_losses, epoch_times

def plot_loss(pytorch_log, jittor_log, save_path='loss_curve.png'):

    pt_loss, pt_times = parse_loss_time(pytorch_log)
    jt_loss, jt_times = parse_loss_time(jittor_log)

    min_len = min(len(pt_loss), len(jt_loss))
    epochs = range(min_len)
    pt_loss = pt_loss[:min_len]
    jt_loss = jt_loss[:min_len]

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(epochs, pt_loss, 'o-', label=f'PyTorch Baseline', color='orangered', linewidth=2)
    ax.plot(epochs, jt_loss, 's--', label=f'Jittor Implementation', color='dodgerblue', linewidth=2)
    ax.set_title('Jittor vs. PyTorch: Training Loss Alignment', fontsize=16)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Average Training Loss', fontsize=12)
    ax.legend(fontsize=12)
    ax.grid(True)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_ylim(0, 1)
    plt.savefig(save_path, dpi=300)
    print(f"The loss alignment curve graph has been saved to: {save_path}")

    # Performance analysis
    avg_pt_time = np.mean(pt_times) if pt_times else 0
    avg_jt_time = np.mean(jt_times) if jt_times else 0

    print(f"The average training time per epoch of PyTorch: {avg_pt_time:.2f} seconds.")
    print(f"The average training time per epoch of Jittor: {avg_jt_time:.2f} seconds.")
    if avg_pt_time > 0 and avg_jt_time > 0:
        print(f"Performance analysis: {(avg_jt_time / avg_pt_time) * 100:.2f}%")

if __name__ == '__main__':
    pytorch_logger = r"./log_rank0.txt" # Replace with your actual PyTorch log path
    # Replace with your actual Jittor log path
    jittor_logger = r"./output/swin_tiny_cats_vs_dogs_jittor_run/swin_tiny_patch4_window7_224/jittor_default/log_rank0.txt"

    plot_alignment_curves(pytorch_logger, jittor_logger)
    plot_loss(pytorch_logger, jittor_logger)