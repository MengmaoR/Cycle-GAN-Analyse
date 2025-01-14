#!/usr/bin/env python3

import re
import matplotlib.pyplot as plt
from collections import defaultdict

def parse_log_line(line):
    pattern = (
       r"Epoch\s+(\d+)/(\d+)\s+\[(\d+)/(\d+)\]\s+--\s+"
       r"loss_G:\s+([\d\.]+)\s+\|\s+"
       r"loss_G_identity:\s+([\d\.]+)\s+\|\s+"
       r"loss_G_GAN:\s+([\d\.]+)\s+\|\s+"
       r"loss_G_cycle:\s+([\d\.]+)\s+\|\s+"
       r"loss_D:\s+([\d\.]+)\s+--\s+ETA:"
    )

    match = re.search(pattern, line)
    if match:
        return {
            'epoch': int(match.group(1)),
            'epoch_total': int(match.group(2)),
            'batch': int(match.group(3)),
            'batch_total': int(match.group(4)),
            'loss_G': float(match.group(5)),
            'loss_G_identity': float(match.group(6)),
            'loss_G_GAN': float(match.group(7)),
            'loss_G_cycle': float(match.group(8)),
            'loss_D': float(match.group(9))
        }
    else:
        return None

def process_log_file(log_file):
    epoch_stats = defaultdict(lambda: {
        'loss_G': 0.0,
        'loss_G_identity': 0.0,
        'loss_G_GAN': 0.0,
        'loss_G_cycle': 0.0,
        'loss_D': 0.0,
        'count': 0
    })

    with open(log_file, 'r') as f:
        for line in f:
            parsed = parse_log_line(line)
            if parsed:
                e = parsed['epoch']
                epoch_stats[e]['loss_G'] += parsed['loss_G']
                epoch_stats[e]['loss_G_identity'] += parsed['loss_G_identity']
                epoch_stats[e]['loss_G_GAN'] += parsed['loss_G_GAN']
                epoch_stats[e]['loss_G_cycle'] += parsed['loss_G_cycle']
                epoch_stats[e]['loss_D'] += parsed['loss_D']
                epoch_stats[e]['count'] += 1

    epochs = sorted(epoch_stats.keys())
    mean_loss_G = []
    mean_loss_G_identity = []
    mean_loss_G_GAN = []
    mean_loss_G_cycle = []
    mean_loss_D = []

    for e in epochs:
        count = epoch_stats[e]['count']
        mean_loss_G.append(epoch_stats[e]['loss_G'] / count)
        mean_loss_G_identity.append(epoch_stats[e]['loss_G_identity'] / count)
        mean_loss_G_GAN.append(epoch_stats[e]['loss_G_GAN'] / count)
        mean_loss_G_cycle.append(epoch_stats[e]['loss_G_cycle'] / count)
        mean_loss_D.append(epoch_stats[e]['loss_D'] / count)

    return epochs, mean_loss_G, mean_loss_G_identity, mean_loss_G_GAN, mean_loss_G_cycle, mean_loss_D

def main():
    log_files = ["log/typical_log.txt", "log/attention_log.txt"]  # 添加你的日志文件路径
    log_names = ["typical", "attention"]  # 日志文件对应的名称
    all_epochs = []
    all_mean_loss_G = []
    all_mean_loss_G_identity = []
    all_mean_loss_G_GAN = []
    all_mean_loss_G_cycle = []
    all_mean_loss_D = []

    for log_file in log_files:
        epochs, mean_loss_G, mean_loss_G_identity, mean_loss_G_GAN, mean_loss_G_cycle, mean_loss_D = process_log_file(log_file)
        all_epochs.append(epochs)
        all_mean_loss_G.append(mean_loss_G)
        all_mean_loss_G_identity.append(mean_loss_G_identity)
        all_mean_loss_G_GAN.append(mean_loss_G_GAN)
        all_mean_loss_G_cycle.append(mean_loss_G_cycle)
        all_mean_loss_D.append(mean_loss_D)

    # 绘制对比图
    plt.figure(figsize=(10,6))
    for i, epochs in enumerate(all_epochs):
        plt.plot(epochs, all_mean_loss_G[i], label=log_names[i])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Average loss_G per Epoch')
    plt.legend()
    plt.grid(True)
    plt.savefig('plot/loss_G_comparison.png', dpi=150)
    plt.close()

    plt.figure(figsize=(10,6))
    for i, epochs in enumerate(all_epochs):
        plt.plot(epochs, all_mean_loss_G_identity[i], label=log_names[i])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Average loss_G_identity per Epoch')
    plt.legend()
    plt.grid(True)
    plt.savefig('plot/loss_G_identity_comparison.png', dpi=150)
    plt.close()

    plt.figure(figsize=(10,6))
    for i, epochs in enumerate(all_epochs):
        plt.plot(epochs, all_mean_loss_G_GAN[i], label=log_names[i])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Average loss_G_GAN per Epoch')
    plt.legend()
    plt.grid(True)
    plt.savefig('plot/loss_G_GAN_comparison.png', dpi=150)
    plt.close()

    plt.figure(figsize=(10,6))
    for i, epochs in enumerate(all_epochs):
        plt.plot(epochs, all_mean_loss_G_cycle[i], label=log_names[i])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Average loss_G_cycle per Epoch')
    plt.legend()
    plt.grid(True)
    plt.savefig('plot/loss_G_cycle_comparison.png', dpi=150)
    plt.close()

    plt.figure(figsize=(10,6))
    for i, epochs in enumerate(all_epochs):
        plt.plot(epochs, all_mean_loss_D[i], label=log_names[i])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Average loss_D per Epoch')
    plt.legend()
    plt.grid(True)
    plt.savefig('plot/loss_D_comparison.png', dpi=150)
    plt.close()

if __name__ == "__main__":
    main()
