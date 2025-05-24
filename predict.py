import argparse
import torch
from model_utils import load_checkpoint
from process import process_image
import json


def predict(image_path, model, device, topk):
    model.to(device)
    model.eval()
    image_tensor = process_image(image_path).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(image_tensor)
        ps = torch.exp(output)
        top_p, top_classes = ps.topk(topk, dim=1)
    idx_to_class = {v: k for k, v in model.class_to_idx.items()}
    top_labels = [idx_to_class[c.item()] for c in top_classes[0]]
    return top_p[0].tolist(), top_labels


def main():
    parser = argparse.ArgumentParser(description='Predict image class.')
    parser.add_argument('image_path', type=str, help='Imagepath')
    parser.add_argument('checkpoint', type=str, help='Checkpoint file')
    parser.add_argument('--top_k', type=int, default=1, help='Return top K most likely classes')
    parser.add_argument('--category_names', type=str, help='cat-to-name file')
    parser.add_argument('--gpu', action='store_true', help='Use GPU if available')

    args = parser.parse_args()

    device = torch.device('cuda' if args.gpu and torch.cuda.is_available() else 'cpu')

    model = load_checkpoint(args.checkpoint)

    probs, classes = predict(args.image_path, model, device, args.top_k)

    if args.category_names:
        with open(args.category_names, 'r') as f:
            cat_to_name = json.load(f)
        class_names = [cat_to_name.get(c, c) for c in classes]
    else:
        class_names = classes

    for i in range(len(class_names)):
        print(f"{class_names[i]}: {probs[i]*100:.2f}%")

if __name__ == '__main__':
    main()
