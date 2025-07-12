"PRODIGY_Trackcode_05"

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt

# Load and preprocess image
def load_image(img_path, max_size=400, shape=None):
    image = Image.open(img_path).convert('RGB')
    
    size = max_size if max(image.size) > max_size else max(image.size)
    in_transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
             std=[0.229, 0.224, 0.225])
    ])
    
    image = transform(image).unsqueeze(0)
    return image.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

# Display image
def imshow(tensor, title=None):
    image = tensor.cpu().clone().squeeze(0)
    image = image * torch.tensor([0.229, 0.224, 0.225]).view(3,1,1) + torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
    image = image.clamp(0, 1)
    plt.imshow(image.permute(1, 2, 0))
    if title: plt.title(title)
    plt.axis('off')
    plt.show()

# Load images
content_path = 'path/to/your/content.jpg'
style_path = 'path/to/your/style.jpg'

content = load_image(content_path)
style = load_image(style_path, shape=content.shape[-2:])

# VGG model for feature extraction
vgg = models.vgg19(pretrained=True).features.eval().to(content.device)

# Freeze parameters
for param in vgg.parameters():
    param.requires_grad_(False)

# Feature layers
def get_features(image, model, layers=None):
    if layers is None:
        layers = {
            '0': 'conv1_1',
            '5': 'conv2_1',
            '10': 'conv3_1',
            '19': 'conv4_1',
            '21': 'conv4_2',  # Content representation
            '28': 'conv5_1'
        }
    features = {}
    x = image
    for name, layer in model._modules.items():
        x = layer(x)
        if name in layers:
            features[layers[name]] = x
    return features

# Gram matrix for style
def gram_matrix(tensor):
    b, c, h, w = tensor.size()
    tensor = tensor.view(c, h * w)
    return torch.mm(tensor, tensor.t())

# Get features
content_features = get_features(content, vgg)
style_features = get_features(style, vgg)

# Compute Gram matrices
style_grams = {layer: gram_matrix(style_features[layer]) for layer in style_features}

# Create target image
target = content.clone().requires_grad_(True)

# Define style weights
style_weights = {
    'conv1_1': 1.0,
    'conv2_1': 0.75,
    'conv3_1': 0.5,
    'conv4_1': 0.2,
    'conv5_1': 0.1
}
content_weight = 1e4
style_weight = 1e2

# Optimizer
optimizer = optim.Adam([target], lr=0.003)

# Training loop
epochs = 300
for i in range(epochs):
    target_features = get_features(target, vgg)
    content_loss = torch.mean((target_features['conv4_2'] - content_features['conv4_2'])**2)
    
    style_loss = 0
    for layer in style_weights:
        target_feature = target_features[layer]
        target_gram = gram_matrix(target_feature)
        style_gram = style_grams[layer]
        layer_style_loss = style_weights[layer] * torch.mean((target_gram - style_gram)**2)
        style_loss += layer_style_loss / (target_feature.shape[1] ** 2)
    
    total_loss = content_weight * content_loss + style_weight * style_loss
    
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    if i % 50 == 0:
        print(f"Epoch {i}/{epochs}, Loss: {total_loss.item():.4f}")

# Show final result
imshow(target, title="Stylized Output")

    image = transform(image).unsqueeze(0)
    return image.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

# Display image
def imshow(tensor, title=None):
    image = tensor.cpu().clone().squeeze(0)
    image = image * torch.tensor([0.229, 0.224, 0.225]).view(3,1,1) + torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
    image = image.clamp(0, 1)
    plt.imshow(image.permute(1, 2, 0))
    if title: plt.title(title)
    plt.axis('off')
    plt.show()

# Load images
content_path = 'path/to/your/content.jpg'
style_path = 'path/to/your/style.jpg'

content = load_image(content_path)
style = load_image(style_path, shape=content.shape[-2:])

# VGG model for feature extraction
vgg = models.vgg19(pretrained=True).features.eval().to(content.device)

# Freeze parameters
for param in vgg.parameters():
    param.requires_grad_(False)

# Feature layers
def get_features(image, model, layers=None):
    if layers is None:
        layers = {
            '0': 'conv1_1',
            '5': 'conv2_1',
            '10': 'conv3_1',
            '19': 'conv4_1',
            '21': 'conv4_2',  # Content representation
            '28': 'conv5_1'
        }
    features = {}
    x = image
    for name, layer in model._modules.items():
        x = layer(x)
        if name in layers:
            features[layers[name]] = x
    return features

# Gram matrix for style
def gram_matrix(tensor):
    b, c, h, w = tensor.size()
    tensor = tensor.view(c, h * w)
    return torch.mm(tensor, tensor.t())

# Get features
content_features = get_features(content, vgg)
style_features = get_features(style, vgg)

# Compute Gram matrices
style_grams = {layer: gram_matrix(style_features[layer]) for layer in style_features}

# Create target image
target = content.clone().requires_grad_(True)

# Define style weights
style_weights = {
    'conv1_1': 1.0,
    'conv2_1': 0.75,
    'conv3_1': 0.5,
    'conv4_1': 0.2,
    'conv5_1': 0.1
}
content_weight = 1e4
style_weight = 1e2

# Optimizer
optimizer = optim.Adam([target], lr=0.003)

# Training loop
epochs = 300
for i in range(epochs):
    target_features = get_features(target, vgg)
    content_loss = torch.mean((target_features['conv4_2'] - content_features['conv4_2'])**2)
    
    style_loss = 0
    for layer in style_weights:
        target_feature = target_features[layer]
        target_gram = gram_matrix(target_feature)
        style_gram = style_grams[layer]
        layer_style_loss = style_weights[layer] * torch.mean((target_gram - style_gram)**2)
        style_loss += layer_style_loss / (target_feature.shape[1] ** 2)
    
    total_loss = content_weight * content_loss + style_weight * style_loss
    
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    if i % 100 == 0:
        print(f"Step {i}/{epochs}, Total Loss: {total_loss.item()}")

# Show final result
final_img = im_convert(target)
plt.imshow(final_img)
plt.axis('off')
plt.show