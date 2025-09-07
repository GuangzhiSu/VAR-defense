import matplotlib.pyplot as plt

FPR = [0.52,0.5,0.61,0.62,0.59,0.46,0.5,0.7,0.49,0.4,0.41,0.49,0.61,0.69,0.38]
TPR= [0.9,0.96,0.92,0.92,0.95,0.89,0.93,0.87,0.9,0.9,0.84,0.9,0.87,0.96,0.84]

layers = [26,27,28,29,30,31,32,33,34,35,36,37,38,39,40]
plt.figure(figsize=(10,6))
plt.plot(layers, TPR, marker='o', label='True Positive Rate (TPR)')
plt.plot(layers, FPR, marker='x', label='False Positive Rate (FPR)')
plt.xlabel('select_layer')
plt.ylabel('Rate')
plt.title('TPR & FPR vs. select_layer')
plt.legend()
plt.grid(True)
plt.savefig('/home/gs285/VAR/my_model/harmfulness_probe/select_layer_sweep_results.png')
plt.show()