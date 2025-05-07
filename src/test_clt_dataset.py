from naive_autoregressive.dataset import CADImageDataset, collate_fn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

dataset = CADImageDataset()
dataloader = DataLoader(
    dataset,
    batch_size=64,
    shuffle=True,
    collate_fn=collate_fn,
)

for i, batch in enumerate(dataloader):
    batch_shapes = {k: v.shape for k, v in batch.items()}
    print(batch_shapes)
    if i == 0:
        for j in range(8):
            plt.imsave(
                f"test_ar_dataset_{j}.png",
                batch["images"][0][j].cpu().numpy(),
            )
    if i == 10:
        break
