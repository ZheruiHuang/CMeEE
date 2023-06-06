import numpy as np
import matplotlib.pyplot as plt

base_lr = 3e-5
max_epoch = 20
warmup_epoch = 2

layers_name = [
    "classifier.layer",
    "bert.encoder.layer5",
    "bert.encoder.layer4",
    "bert.encoder.layer3",
    "bert.encoder.layer2",
    "bert.encoder.layer1",
]
N = len(layers_name)

init_point = np.zeros((N, 2))  # (N, 2)
peak_point = np.zeros((N, 2))  # (N, 2)
peak_point[:, 0] = warmup_epoch
peak_point[:, 1] = base_lr * np.power(0.8, np.arange(N))
end_point = np.zeros((N, 2))  # (N, 2)
end_point[:, 0] = max_epoch
end_point[:, 1] = 0.15 * base_lr * np.power(0.8, np.arange(N))

# N, 3, 2
points = np.concatenate(
    [init_point[:, :, np.newaxis], peak_point[:, :, np.newaxis], end_point[:, :, np.newaxis]], axis=2
)

for i in range(N):
    plt.plot(points[i, 0, :], points[i, 1, :], label=layers_name[i], linewidth=1)
plt.legend()
plt.savefig("../layer-wise-decay.pdf")
