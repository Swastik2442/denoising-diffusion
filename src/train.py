from pathlib import Path

from datasets import load_dataset
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image

from .model import Unet
from .sample import p_losses, sample, timesteps

device = "cuda" if torch.cuda.is_available() else "cpu"
image_size = 64
channels = 3
batch_size = 64
epochs = 50
lr = 1e-3
save_and_sample_every = 1000

results_folder = Path("./results")
results_folder.mkdir(exist_ok = True)

transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Lambda(lambda t: (t * 2) - 1)
])
def transforms_map(examples):
    examples["pixel_values"] = [transform(image) for image in examples["image"]]
    del examples["image"]
    return examples

def num_to_groups(num: int, divisor: int):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

def save_samples(model: torch.nn.Module, step: int):
    milestone = step // save_and_sample_every
    batches = num_to_groups(4, batch_size)
    all_images_list = list(map(lambda n: sample(model, image_size, batch_size=n, channels=channels), batches))
    all_images = torch.cat(all_images_list, dim=0) # type: ignore
    all_images = (all_images + 1) * 0.5
    save_image(all_images, str(results_folder / f'sample-{milestone}.png'), nrow = 6)

model = Unet(
    dim=image_size,
    channels=channels,
    dim_mults=(1, 2, 4,)
)
model.to(device)
optimizer = Adam(model.parameters(), lr=lr)

dataset = load_dataset("korexyz/celeba-hq-256x256")
transformed_dataset = dataset.with_transform(transforms_map).remove_columns("label")
dataloader = DataLoader(transformed_dataset["train"], batch_size=batch_size, shuffle=True)

for epoch in range(epochs):
    for step, batch in enumerate(dataloader):
        optimizer.zero_grad()

        batch_size = batch["pixel_values"].shape[0]
        batch = batch["pixel_values"].to(device)

        t = torch.randint(0, timesteps, (batch_size,), device=device).long()

        loss = p_losses(model, batch, t, loss_type="huber")

        if step % 100 == 0:
            print("Loss:", loss.item())

        loss.backward()
        optimizer.step()

        if step != 0 and step % save_and_sample_every == 0:
            save_samples(model, step)

torch.save(model.state_dict(), "out/simple-model-linear-beta-celebahq-all-20251102-1800.pth")
