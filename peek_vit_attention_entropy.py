import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import pickle
import os
import cv2
from scipy.special import entr

# ----- simple VIT model -----
class PatchEmbedding(nn.Module):
    def __init__(self, img_size=32, patch_size=4, in_channels=3, embed_dim=64):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)  # (B, embed_dim, H/patch, W/patch)
        x = x.flatten(2)  # (B, embed_dim, N)
        x = x.transpose(1, 2)  # (B, N, embed_dim)
        return x

class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim, num_heads, hidden_dim):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim)
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.last_attn_weights = None  # Will be filled by hook

    def forward(self, x):
        attn_out, attn_weights = self.attn(x, x, x, need_weights=True, average_attn_weights=False)
        self.last_attn_weights = attn_weights  # Save attention weights for hook
        x = self.norm1(x + attn_out)
        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)
        return x

class VisionTransformer(nn.Module):
    def __init__(self, img_size=32, patch_size=4, num_classes=10, embed_dim=64, depth=4, num_heads=4, hidden_dim=128):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, 3, embed_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, (img_size // patch_size)**2, embed_dim))
        self.encoder = nn.ModuleList([
            TransformerEncoder(embed_dim, num_heads, hidden_dim)
            for _ in range(depth)
        ])
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, num_classes)
        )

    def forward(self, x):
        x = self.patch_embed(x)
        x = x + self.pos_embed
        for layer in self.encoder:
            x = layer(x)
        x = x.mean(dim=1)
        x = self.mlp_head(x)
        return x

# ----- attention capture -----
feature_maps = {}

def register_hooks(model):
    for idx, module in enumerate(model.encoder):
        layer_name = f"EncoderLayer_{idx}"
        def hook_fn(m, i, o, name=layer_name):
            if hasattr(m, 'last_attn_weights'):
                feature_maps[name] = m.last_attn_weights.detach().cpu()
        module.register_forward_hook(hook_fn)

# ----- attention PEEK -----
def compute_attention_PEEK(attn_weights, h, w):
    """
    attn_weights: [1, num_heads, N, N] → squeeze batch
    """
    if isinstance(attn_weights, np.ndarray):
        attn_weights = torch.tensor(attn_weights)
    
    print(f"[DEBUG] attn_weights shape: {attn_weights.shape}")  # Expect [num_heads, N, N]

    if attn_weights.dim() == 4:
        attn_weights = attn_weights.squeeze(0)  # remove batch dimension

    print(f"[DEBUG] attn_weights shape: {attn_weights.shape}")  # Expect [num_heads, N, N]

    num_heads, N, _ = attn_weights.shape
    attn_mean = attn_weights.mean(dim=0).numpy()  # [N, N]
    entropy = entr(attn_mean).sum(axis=-1)        # [N]
    print(f"[DEBUG] entropy shape: {entropy.shape}")

    side = int(np.sqrt(N))
    if side * side != N:
        raise ValueError(f"Cannot reshape {N} tokens into square grid (expected perfect square).")

    entropy_map = entropy.reshape(side, side)
    peek_map = cv2.resize(entropy_map, (w, h), interpolation=cv2.INTER_CUBIC)
    return peek_map


def save_feature_maps(model, feature_maps, sample_image, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with torch.no_grad():
        model.eval()
        _ = model(sample_image)
        with open(save_path, 'wb') as f:
            pickle.dump({k: v.numpy() for k, v in feature_maps.items()}, f)
    print(f"Feature maps saved at {save_path}")
    return save_path

def plot_PEEK(modules, sample_image, feature_map_path):
    if not os.path.exists(feature_map_path):
        print(f"Feature map path {feature_map_path} does not exist.")
        return

    image = sample_image.squeeze().cpu().numpy()
    if image.ndim == 3 and image.shape[0] == 3:
        image = np.transpose(image, (1, 2, 0))
    h, w = image.shape[:2]

    with open(feature_map_path, 'rb') as f:
        loaded_feature_maps = pickle.load(f)

    fig, axes = plt.subplots(len(modules), 2, figsize=(8, 4 * len(modules)))

    for i, layer in enumerate(modules):
        layer_name = f"EncoderLayer_{i}"

        axes[i, 0].imshow(image)
        axes[i, 0].set_title("Input Image")
        axes[i, 0].axis('off')

        attn_weights = loaded_feature_maps.get(layer_name)
        if attn_weights is None:
            raise KeyError(f"Layer {layer_name} not found in feature maps")

        peek_map = compute_attention_PEEK(torch.tensor(attn_weights), h, w)
        axes[i, 1].imshow(image)
        axes[i, 1].imshow(peek_map, alpha=0.6, cmap='jet')
        axes[i, 1].set_title(f"PEEK - {layer_name}")
        axes[i, 1].axis('off')

    fig.tight_layout()
    plt.show()

# ------ MAIN FUNCTION -------
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor()
    ])

    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    model = VisionTransformer(img_size=32, patch_size=4, num_classes=10).to(device)
    register_hooks(model)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Train loop
    model.train()
    for epoch in range(2):
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

    # Pick one test image
    sample_image, _ = test_dataset[0]
    sample_image = sample_image.unsqueeze(0).to(device)

    save_path = './features/sample_image_attn.pkl'
    feature_map_path = save_feature_maps(model, feature_maps, sample_image, save_path)

    modules = [m for m in model.encoder]
    plot_PEEK(modules, sample_image, feature_map_path)

main()
