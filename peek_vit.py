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
        self.attn = nn.MultiheadAttention(embed_dim, num_heads)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim)
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        attn_out, _ = self.attn(x, x, x)
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

# ----- initializations for capturing feature maps -----
feature_maps = {}

def hook_fn(m, i, o):
    if not m.training:
        if isinstance(o, tuple):
            o = o[0]
        print(f"Forward Hook - {m.__class__.__name__}: Output Shape {o.shape}")
        feature_maps[str(m)] = o  # <-- SAVE layer name (string) instead of object

def register_hooks(model):
    for module in model.modules():
        if isinstance(module, nn.MultiheadAttention):
            module.register_forward_hook(hook_fn)

# ----- PEEK functions (exact copy of CNN) -----
def compute_PEEK(feature_maps, h, w):
    positivized_maps = feature_maps + np.abs(np.min(feature_maps))
    entropy_map = -np.sum(entr(positivized_maps), axis=-1)
    peek_map = cv2.resize(entropy_map, (w, h))
    return peek_map

def save_feature_maps(model, feature_maps, sample_image, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with torch.no_grad():
        model.eval()
        _ = model(sample_image)
        with open(save_path, 'wb') as f:
            pickle.dump({layer: fmap.cpu().numpy() for layer, fmap in feature_maps.items()}, f)
    print(f"Feature maps saved at {save_path}")
    return save_path

def plot_PEEK(modules, sample_image, feature_map_path):
    feature_map_path = feature_map_path if os.path.exists(feature_map_path) else None
    if feature_map_path is None:
        print(f"Feature map path {feature_map_path} does not exist. Please run save_feature_maps first.")

    # Load original image
    # image = sample_image.squeeze().cpu().numpy()
    # #####h, w = image.shape
    # #####_, _, h, w = image.shape
    # _, h, w = image.shape
    
    # load original image
    image = sample_image.squeeze().cpu().numpy()

    # fixing dimension order if needed
    if image.ndim == 3:
        if image.shape[0] == 3:  # (3, H, W) --> (H, W, 3)
            image = np.transpose(image, (1, 2, 0))
        h, w = image.shape[:2]
    elif image.ndim == 2:
        h, w = image.shape
    else:
        raise ValueError(f"Unexpected image shape: {image.shape}")
    
    # load feature maps
    with open(feature_map_path, 'rb') as f:
        loaded_feature_maps = pickle.load(f)

    # plot for each convolutional layer
    fig, axes = plt.subplots(len(modules), 2, figsize=(8, 4 * len(modules)))

    for i, layer in enumerate(modules):
        # convert layer object to string to match saved keys
        layer_name = str(layer)

        # original image
        axes[i, 0].imshow(image, cmap='gray')
        axes[i, 0].set_title('Input')
        axes[i, 0].axis('off')

        # retrieve the feature maps using the layer name (as string)
        feature_maps = loaded_feature_maps.get(layer_name, None)
        if feature_maps is None:
            raise KeyError(f"Layer {layer_name} not found in loaded_feature_maps")

        feature_maps = feature_maps[0]  # Access the first element
        feature_maps = np.moveaxis(feature_maps, 0, -1)  # Rearrange channels
        peek_map = compute_PEEK(feature_maps, h, w)  # Compute PEEK map

        # plot PEEK map overlaid on the original image
        axes[i, 1].imshow(image, cmap='gray')  # Original image
        axes[i, 1].imshow(peek_map, alpha=0.7, cmap='jet')  # Overlay PEEK
        axes[i, 1].set_title(f'PEEK - {layer_name}')
        axes[i, 1].axis('off')

    fig.tight_layout()
    plt.show()


# ----- Main Runner -----
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load CIFAR-10
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor()
    ])
    print("hello loaded")

    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    print("hello train/test setup")

    # Model
    model = VisionTransformer(img_size=32, patch_size=4, num_classes=10).to(device)
    register_hooks(model)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    print("hello model init")

    # Train
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
    
    print("hello trained")

    # Select Sample Image
    sample_image, _ = test_dataset[0]
    sample_image = sample_image.unsqueeze(0).to(device)
    print("hello sample image selected")

    # Save feature maps
    save_path = './features/sample_image.pkl'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    print("hello sample feature map saved")
    feature_map_path = save_feature_maps(model, feature_maps, sample_image, save_path)

    # Plot PEEK
    modules = [m for m in model.modules() if isinstance(m, nn.MultiheadAttention)] # only saving attention modules 
    plot_PEEK(modules, sample_image, feature_map_path)
    print("plotted")

main()