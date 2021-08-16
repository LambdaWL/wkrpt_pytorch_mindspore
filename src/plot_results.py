import matplotlib.pyplot as plt
import numpy as np


torch_losses = [
    1.1203, 0.6851, 0.6341, 0.6247, 0.6173, 
    0.6116, 0.6004, 0.5873, 0.5819, 0.5778, 
    0.5734, 0.5683, 0.5631, 0.5585, 0.5548, 
    0.5507, 0.5464, 0.5424, 0.5375, 0.5333,
]
ms_losses = [
    1.1615, 0.6861, 0.6337, 0.626, 0.6181, 
    0.6154, 0.606, 0.5935, 0.5826, 0.5756, 
    0.5706, 0.5651, 0.5604, 0.5558, 0.5517, 
    0.5475, 0.5429, 0.537, 0.5317, 0.5269,
]
epochs = np.arange(1, 21).astype(np.int32)

plt.xlabel("Epoch", fontdict={"family": "Times New Roman", "size": 13})
plt.ylabel("Train Loss", fontdict={"family": "Times New Roman", "size": 13})
plt.xticks(np.arange(1, 21, 2).astype(np.int32), fontproperties="Times New Roman", size=12)
plt.plot(epochs, torch_losses, label="PyTorch")
plt.plot(epochs, ms_losses, label="Mindspore")
plt.legend(prop={"family": "Times New Roman", "size": 10})
plt.savefig("loss_vs_epoch.jpg", bbox_inches="tight")
plt.show()

torch_times = [
    4.61, 4.59, 5.26, 4.36, 4.56, 
    4.39, 4.39, 4.38, 4.32, 4.38, 
    4.38, 4.33, 4.38, 4.31, 4.37, 
    4.32, 4.33, 4.32, 4.27, 4.36
]
ms_times = [
    80.13, 23.9, 23.51, 22.4, 22.4, 
    21.5, 23.39, 22.51, 22.19, 22.91, 
    22.31, 22.7, 23.29, 22.51, 22.09, 
    22.01, 21.8, 23.51, 22.79, 22.2
]

print(f"PyTorch Time: first={torch_times[0]}s, avg={sum(torch_times)/20}s")
print(f"Mindspore Time: first={ms_times[0]}s, avg={sum(ms_times)/20}s")
