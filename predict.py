from common import *

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    image = Image.open(image).convert("RGB")

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

def predict(image_path, model, device, topk):
    model.eval()
    
    # Ensure the image is a FloatTensor and on the correct device
    image_tensor = process_image(image_path).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(image_tensor)
        ps = torch.exp(output)
        top_p, top_classes = ps.topk(topk, dim=1)
    
    idx_to_class = {v: k for k, v in model.class_to_idx.items()}
    top_labels = [idx_to_class[c.item()] for c in top_classes[0]]
    
    return top_p[0].tolist(), top_labels


def load_checkpoint(filename):
    checkpoint = torch.load(filename, map_location='cpu')

    arch = checkpoint.get('arch', 'vgg16_bn')
    model=getModel(arch)

    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    return model


def main(image_path, checkpoint, top_k, category_names, gpu):

    device = torch.device('cuda' if args.gpu and torch.cuda.is_available() else 'cpu')
    model = load_checkpoint(checkpoint)
    model.to(device)

    probs, classes = predict(image_path, model, device, top_k)

    if category_names:
        with open(args.category_names, 'r') as f:
            cat_to_name = json.load(f)
        class_names = [cat_to_name.get(c, c) for c in classes]
    else:
        class_names = classes

    for i in range(len(class_names)):
        print(f"{class_names[i]}: {probs[i]*100:.2f}%")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict image class')
    parser.add_argument('image_path', type=str, help='Image Path')
    parser.add_argument('checkpoint', type=str, default=".", help='Checkpoint directory (filename is checkpoint.pth)')
    parser.add_argument('--top_k', type=int, default=5, help='Return top K most likely classes')
    parser.add_argument('--category_names', type=str, default='cat_to_name.json', help='cat_to_name.json file')
    parser.add_argument('--gpu', action='store_true', help='Use GPU if available')

    args = parser.parse_args()

    main(args.image_path, args.checkpoint, args.top_k, args.category_names, args.gpu)
