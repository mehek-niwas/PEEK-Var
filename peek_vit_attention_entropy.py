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
    def __init__(self, img_size=32, patch_size=4, num_classes=10, embed_dim=64, depth=6, num_heads=4, hidden_dim=128):
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
def compute_attention_PEEK(layer_name, attn_weights, h, w, save_entropy=True, return_entropy=True):  # MUST BE CALLED FOR EVERY LAYER SPECIFIED. default returns entropy & peek_map
                                                                                # SAVES NUMPY ARRAY OF ENTROPY IN AN ENTROPY DIRECTORY
    """
    attn_weights: [1, num_heads, N, N] → squeeze batch
    """
    if isinstance(attn_weights, np.ndarray):
        attn_weights = torch.tensor(attn_weights)
    
    print(f"[DEBUG] attn_weights shape: {attn_weights.shape}")  # [num_heads, N, N] --> 4, 64, 64
    
    # confirming softmax sum = 1 (to verify PMF)
    # row_sums = attn_weights.sum(dim=-1)  # sum over key_len
    # print(row_sums)

    if attn_weights.dim() == 4:
        attn_weights = attn_weights.squeeze(0)  # remove batch dimension

    print(f"[DEBUG] attn_weights shape: {attn_weights.shape}")  # [num_heads, N, N] ---> 1, 4, 64, 64

    num_heads, N, _ = attn_weights.shape
    attn_mean = attn_weights.mean(dim=0).numpy()  # [N, N]  ---> 64, 64 (this is the mean attention matrix that results from all the heads)
    entropy = entr(attn_mean).sum(axis=-1)        # [N] --> 64 (this is the entropy of the mean attention weights for each token) --> 1 entropy value per token for the entire layer
    entropy_var = entropy.var() # might have to do np.var
    entropy_avg = entropy.mean()
    entropy_min = entropy.min()
    entropy_max = entropy.max()

    if save_entropy:
        entropy = entropy.cpu().numpy() if torch.is_tensor(entropy) else entropy
        os.makedirs("entropies", exist_ok=True)
        np.save(f"entropies/entropy_{layer_name}.npy", entropy)

    # # STRAIGHT FROM GPT.open()
    # entropy = entropy.cpu().numpy() if torch.is_tensor(entropy) else entropy
    # # Create plot
    # plt.boxplot(entropy)
    # plt.title("Entropy Distribution")
    # plt.ylabel("Entropy")
    # plt.grid(True)
    # # Save with a custom filename
    # filename = f"entropy_boxplot_{layer_name}.png"
    # plt.savefig(filename)
    # plt.close()
    # print(f"Saved plot as {filename}")
    # # Save with a custom filename
    # filename = f"entropy_boxplot_{layer_name}.png"
    # plt.savefig(filename)
    # plt.close()
    # print(f"Saved plot as {filename}")
    # STRAIGHT FROM GPT.close()
    
    print("entropy token distribution information. entropy was calculate from the mean attention weight across all heads [for whichever layer this is right now]")
    
    print(f"[DEBUG] entropy shape: {entropy.shape}") # ---> 64
    print(f"[INSPECT] entropy variance: {entropy_var}") # should only be one number
    print(f"[INSPECT] entropy average: {entropy_avg}") # should only be one number
    print(f"[INSPECT] entropy min: {entropy_min}") # should only be one number
    print(f"[INSPECT] entropy min: {entropy_max}") # should only be one number

    side = int(np.sqrt(N)) # --> sqrt(64) = 8
    
    entropy_map = entropy.reshape(side, side) # --> 8 x 8
    peek_map = cv2.resize(entropy_map, (w, h), interpolation=cv2.INTER_CUBIC) 
    
    if (return_entropy):
        return entropy, peek_map
    else:
        return peek_map


def save_feature_maps_and_predict(model, classes, feature_maps, sample_image, true_label, save_path):
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model.eval()
    with torch.no_grad():
        output = model(sample_image)
        predicted_class_id = output.argmax(dim=1).item() # i can also get the next few closest classes 
        predicted_class = classes[predicted_class_id]
        true_class = classes[true_label]
        with open(save_path, 'wb') as f:
            pickle.dump({k: v.numpy() for k, v in feature_maps.items()}, f)
    print(f"Feature maps saved at {save_path}")
    
    #### temporary code for correct / incorrect. this is not currently stored in a variable
    print(f"Predicted: {predicted_class} (index {predicted_class_id})")
    print(f"Actual: {true_class} (index {true_label})")
    print("Correct!" if predicted_class_id == true_label else "Incorrect.")
    ####

    return save_path, predicted_class


def plot_PEEK(modules, sample_image, feature_map_path):  #       THIS IS CALLED FOR A SAMPLE IMAGE AFTER ITS RESPECTIVE FEATURE MAPS HAVE BEEN SAVED.
                                                         #       THIS FUNCTION ALSO CALLS THE COMPUTE_ATTENTION_PEEK METHOD FOR EACH FEATURE MAP OF THE IMAGE
                                                         #        RETURNS GIGA PLOT WITH SUBPLOTS FOR EACH LAYER W/ ENTROPIC HEAT MAP OF IMAGE
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

        entropy, peek_map = compute_attention_PEEK(layer_name, torch.tensor(attn_weights), h, w)
        axes[i, 1].imshow(image)
        axes[i, 1].imshow(peek_map, alpha=0.6, cmap="jet")
        axes[i, 1].set_title(f"PEEK - {layer_name}")
        axes[i, 1].axis("off")

    fig.tight_layout()
    
    plt.savefig("image_entropy_heatmap_plot.png")
    plt.show()
    plt.close()
    


# ------ MAIN FUNCTION -------
def main():
    print("hello beta")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor()
    ])

    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform) # 50000 training images
    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform) # 10000 testing images
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    model = VisionTransformer(img_size=32, patch_size=4, num_classes=10).to(device)
    register_hooks(model)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Train loop
    model.train()
    for epoch in range(7):
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
    
    # get classes 
    classes = test_dataset.classes  # ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    # Pick one test image
    sample_image, true_label = test_dataset[2]
    sample_image = sample_image.unsqueeze(0).to(device)
    # Save feature maps
    save_path = './features/sample_image_attn.pkl'
    
    feature_map_path, prediction = save_feature_maps_and_predict(model, classes, feature_maps, sample_image, true_label, save_path) # model is set to evaluation mode in this function
    
    modules = [m for m in model.encoder]
    plot_PEEK(modules, sample_image, feature_map_path)
    print(prediction)
    
    print("\n model evaluation summary (on test dataset)")
    
    ## STRAIGHT FROM GPT AYYYEEEEE.open()
    from collections import Counter
    import torch.nn.functional as F

    model.eval()
    correct = 0
    total = 0
    confusion_pairs = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Log confusion pairs
            for p, t in zip(predicted.cpu(), labels.cpu()):
                if p != t:
                    confusion_pairs.append((t.item(), p.item()))

    accuracy = 100 * correct / total
    print(f"\nTest Accuracy: {accuracy:.2f}%")
    confusion_counts = Counter(confusion_pairs)
    most_common_confusions = confusion_counts.most_common(5) # i did not know most_common is a command

    print("\nMost common confusions:")
    for (true_id, pred_id), count in most_common_confusions:
        print(f"  {classes[true_id]} → ❌ {classes[pred_id]}: {count} times")
    # STRAIGHT FROM GPT AYEEEE.close()

main()