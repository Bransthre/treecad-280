from naive_autoregressive.dataset import AutoRegressiveDataset
import matplotlib.pyplot as plt

dataset = AutoRegressiveDataset(batch_size=4, num_renders=10)
batch = dataset._produce_batch()
print({k: v.shape for k, v in batch.items()})


plt.imsave(
    "test_ar_dataset.png",
    batch["renderings"][0, 0].cpu().numpy(),
)
plt.imsave(
    "test_ar_dataset_2.png",
    batch["renderings"][0, 1].cpu().numpy(),
)
plt.imsave(
    "test_ar_dataset_3.png",
    batch["renderings"][0, 2].cpu().numpy(),
)
