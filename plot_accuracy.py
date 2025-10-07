#!/usr/bin/env python3
import matplotlib.pyplot as plt
import re

# 从日志文件中提取准确率数据
def extract_accuracy_data(log_file):
    accuracies = []
    epochs = []
    
    with open(log_file, 'r') as f:
        for line in f:
            if 'Accm:' in line and 'Ep]:' in line:
                # 提取epoch信息
                epoch_match = re.search(r'\[Ep\]: \[\s*(\d+)/100\]', line)
                if epoch_match:
                    epoch = int(epoch_match.group(1))
                    
                    # 提取准确率
                    acc_match = re.search(r'Accm: ([0-9.]+)', line)
                    if acc_match:
                        acc = float(acc_match.group(1))
                        accuracies.append(acc)
                        epochs.append(epoch)
    
    return epochs, accuracies

# 提取数据
epochs, accuracies = extract_accuracy_data('/home/gs285/VAR-defense/slurm-118380.out')

# 创建图表
plt.figure(figsize=(12, 6))
plt.plot(epochs, accuracies, 'b-', linewidth=2, marker='o', markersize=3)
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('Training Accuracy Progress')
plt.grid(True, alpha=0.3)
plt.ylim(50, 75)

# 添加统计信息
final_acc = accuracies[-1] if accuracies else 0
initial_acc = accuracies[0] if accuracies else 0
improvement = final_acc - initial_acc

plt.text(0.02, 0.98, f'Initial Accuracy: {initial_acc:.1f}%\nFinal Accuracy: {final_acc:.1f}%\nImprovement: +{improvement:.1f}%', 
         transform=plt.gca().transAxes, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

plt.tight_layout()
plt.savefig('/home/gs285/VAR-defense/accuracy_progress.png', dpi=300, bbox_inches='tight')
print(f"Accuracy plot saved to accuracy_progress.png")
print(f"Training completed: {len(accuracies)} data points")
print(f"Accuracy improved from {initial_acc:.1f}% to {final_acc:.1f}% (+{improvement:.1f}%)")
