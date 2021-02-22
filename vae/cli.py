import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils import data
from torchvision.utils import save_image
from pathlib import Path
import fire
from vae import VAE

def train_from_folder(
    data_dir = "../data/",
    results_dir = "../data/results/",
    models_dir = "../data/models",
    image_size = 28,
    batch_size = 256,
    num_epochs = 15,
    learning_rate = 1e-3,
    hidden_dim = 400,
    latent_dim = 20,
    save_every = 10,
    seed = 42,
    amp = False,
    device = "cuda"
):
    data_dir = Path(data_dir)
    sample_dir = Path(results_dir)
    models_dir = Path(models_dir)
    sample_dir.mkdir(exist_ok=True)
    models_dir.mkdir(exist_ok=True)

    device = torch.device(device)

    dataset = datasets.MNIST(root=data_dir,
                             train=True,
                             transform=transforms.ToTensor(),
                             download=True)

    dataloader = data.DataLoader(dataset=dataset,
                                 batch_size=batch_size,
                                 shuffle=True)

    model = VAE(image_size**2, hidden_dim, latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        for i, (x, _) in enumerate(dataloader):
            x = x.to(device).view(-1, image_size ** 2)
            x_reconstructed, mu, log_var = model(x)

            reconst_loss = F.binary_cross_entropy(x_reconstructed, x, size_average=False)
            kl_div = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

            total_loss = reconst_loss + kl_div
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            if (i + 1) % 10 == 0:
                print(f"Epoch [{epoch + 1}/{num_epochs}], Steps [{i + 1}/{len(dataloader)}], "
                      f"Reconstruction loss: {reconst_loss.item():.4f}, KL div: {kl_div.item():.4f}")

        with torch.no_grad():
            z = torch.randn(batch_size, latent_dim).to(device)
            out = model.decode(z).view(-1, 1, 28, 28)
            save_image(out, Path(sample_dir) / f"sampled-{epoch + 1}.png")

            out, _, _ = model(x)
            x_concat = torch.cat([x.view(-1, 1, 28, 28), out.view(-1, 1, 28, 28)], dim=3)
            save_image(x_concat, Path(sample_dir) / f"reconst-{epoch + 1}.png")

def main():
    fire.Fire(train_from_folder)