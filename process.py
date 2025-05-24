from PIL import Image
import numpy as np
import torch

def process_image(image_path):
    """Scales, crops, and normalizes a PIL image for a PyTorch model, returns a FloatTensor."""
    image = Image.open(image_path).convert("RGB")

    # Resize keeping aspect ratio so that shortest side = 256
    aspect = image.width / image.height
    if image.width < image.height:
        image = image.resize((256, int(256 / aspect)))
    else:
        image = image.resize((int(256 * aspect), 256))

    # Center crop 224x224
    left = (image.width - 224) / 2
    top = (image.height - 224) / 2
    right = left + 224
    bottom = top + 224
    image = image.crop((left, top, right, bottom))

    # Convert to NumPy and normalize
    np_image = np.array(image) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - mean) / std

    # Reorder dimensions
    np_image = np_image.transpose((2, 0, 1))
    return torch.from_numpy(np_image).type(torch.FloatTensor)


def imshow(image_tensor, ax=None, title=None):
    if ax is None:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()

    image = image_tensor.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    image = np.clip(image, 0, 1)

    ax.imshow(image)
    if title:
        ax.set_title(title)
    ax.axis('off')
    return ax
