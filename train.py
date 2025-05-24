import argparse
import torch
from model_utils import build_model, save_checkpoint, load_data
import datetime

torch.backends.cudnn.benchmark = True


TRAIN = "train"
VALID = "validate"
TEST = "test"


def train_model(model, dataloaders, image_datasets, criterion, optimizer, epochs, device):
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    for epoch in range(epochs):
        print(f"{datetime.datetime.now()} Running epoch #{epoch+1}/{epochs}")
        train_loss = 0
        model.train()
        for inputs, labels in dataloaders[TRAIN]:
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0
        accuracy = 0
        with torch.no_grad():
            for inputs, labels in dataloaders[VALID]:
                inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                outputs = model(inputs)
                val_loss += criterion(outputs, labels).item()
                _, preds = torch.max(outputs, 1)
                accuracy += torch.sum(preds == labels.data)

        scheduler.step()

        print(f"{datetime.datetime.now()} Summary of Epoch {epoch+1}/{epochs}.. ",
              f"Train Loss: {train_loss/len(dataloaders[TRAIN]):.3f}.. ",
              f"Validation Loss: {val_loss/len(dataloaders[VALID]):.3f}.. ",
              f"Validation Accuracy: {accuracy.double()/len(image_datasets[VALID]):.3f}")
        

def main():
    parser = argparse.ArgumentParser(description='Train an image classifier.')
    parser.add_argument('data_dir', type=str, help='Path to dataset')
    parser.add_argument('--save_dir', type=str, default='.', help='Directory to save checkpoints')
    parser.add_argument('--arch', type=str, default='vgg16_bn', help='Model architecture (vgg13, vgg16, vgg16_bn)')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--hidden_units', type=int, default=4096, help='Hidden layer units')
    parser.add_argument('--epochs', type=int, default=5, help='Number of training epochs')
    parser.add_argument('--gpu', action='store_true', help='Use GPU if available')

    args = parser.parse_args()

    device = torch.device('cuda' if args.gpu and torch.cuda.is_available() else 'cpu')

    dataloaders, image_datasets = load_data(args.data_dir)
    model, criterion, optimizer = build_model(args.arch, args.hidden_units, args.learning_rate)

    model.to(device)

    train_model(model, dataloaders, image_datasets, criterion, optimizer, args.epochs, device)

    save_path = f"{args.save_dir}/checkpoint.pth"
    save_checkpoint(model, image_datasets[TRAIN].class_to_idx, save_path, arch=args.arch)


if __name__ == '__main__':
    main()
