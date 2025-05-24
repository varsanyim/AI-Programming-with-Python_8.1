from common import *

torch.backends.cudnn.benchmark = True

def buildModel(arch, hidden_units, device):
    print(">> Init Model")
    model=getModel(arch)
    for param in model.parameters():
        param.requires_grad = False

    input_features = model.classifier[0].in_features if arch.startswith("vgg") else model.fc.in_features
    
    classifier = nn.Sequential(
        nn.Linear(input_features, hidden_units),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(hidden_units, 102),
        nn.LogSoftmax(dim=1)
    )

    if arch == "resnet18":
        model.fc = classifier
#         for param in model.fc.parameters():
#             param.requires_grad = True
    else:
        model.classifier = classifier
#         for param in model.classifier.parameters():
#             param.requires_grad = True
        
    model.to(device)
    
    return model, classifier


def load_data(data_dir):
    print(">> Load Data")
    train_dir = os.path.join(data_dir, TRAIN)
    valid_dir = os.path.join(data_dir, VALID)
    test_dir  = os.path.join(data_dir, TEST)

    data_transforms = {
        TRAIN: transforms.Compose([
            transforms.RandomResizedCrop(SIZE),
            transforms.RandomRotation(30),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(t_mean, t_std)
        ]),
        TEST: transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(SIZE),
            transforms.ToTensor(),
            transforms.Normalize(t_mean, t_std)
        ]),
        VALID: transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(SIZE),
            transforms.ToTensor(),
            transforms.Normalize(t_mean, t_std)
        ])
    }

    image_datasets = {
        TRAIN: datasets.ImageFolder(train_dir, transform=data_transforms[TRAIN]),
        VALID: datasets.ImageFolder(valid_dir, transform=data_transforms[VALID]),
        TEST: datasets.ImageFolder(test_dir, transform=data_transforms[TEST])
    }

    dataloaders = {
        TRAIN: DataLoader(image_datasets[TRAIN], batch_size=BATCH_SIZE, shuffle=True, num_workers=WORKERS, pin_memory=True),
        VALID: DataLoader(image_datasets[VALID], batch_size=BATCH_SIZE, shuffle=True, num_workers=WORKERS, pin_memory=True),
        TEST: DataLoader(image_datasets[TEST], batch_size=BATCH_SIZE, shuffle=True, num_workers=WORKERS, pin_memory=True)
    }
    
    return image_datasets, dataloaders

def train(model, epochs, dataloaders, image_datasets, optimizer, device):
    print(">> Train")
    criterion = nn.NLLLoss()
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
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

        # Validation loop
        model.eval()
        val_loss = 0
        accuracy = 0
        with torch.no_grad():
            for inputs, labels in dataloaders[VALID]:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                val_loss += criterion(outputs, labels).item()
                _, preds = torch.max(outputs, 1)
                accuracy += torch.sum(preds == labels.data)
        scheduler.step()

        print(f"{datetime.datetime.now()} Summary of Epoch {epoch+1}/{epochs}.. ",
              f"Train Loss: {train_loss/len(dataloaders[TRAIN]):.3f}.. ",
              f"Validation Loss: {val_loss/len(dataloaders[VALID]):.3f}.. ",
              f"Validation Accuracy: {accuracy.double()/len(image_datasets[VALID]):.3f}")

    
def main(data_dir, checkpoint, arch, learning_rate, hidden_units, epochs, gpu):
    try:
        device = torch.device('cuda' if gpu and torch.cuda.is_available() else 'cpu')
        print(device)

        model, classifier = buildModel(arch, hidden_units, device)
        trainable_params = filter(lambda p: p.requires_grad, model.parameters())
        optimizer = optim.Adam(trainable_params, lr=learning_rate)

        image_datasets, dataloaders = load_data(data_dir)

        train(model, epochs, dataloaders, image_datasets, optimizer, device)

        save_checkpoint(model, image_datasets[TRAIN].class_to_idx, checkpoint, arch)
    except Exception as e:
        print(e)
    print("So Long, and Thanks for All the Fish")

# --------------------------------------------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train an image classifier.')
    parser.add_argument('data_dir', type=str, help='Path to dataset')
    parser.add_argument('--checkpoint', type=str, default='checkpoint.pth', help='checkpoint filename to save. ')
    parser.add_argument('--arch', type=str, default='vgg16_bn', help='Model architecture (vgg13, vgg16, vgg16_bn)')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--hidden_units', type=int, default=4096, help='Hidden layer units')
    parser.add_argument('--epochs', type=int, default=5, help='Number of training epochs')
    parser.add_argument('--gpu', action='store_true', help='Use GPU if available')
  
    args = parser.parse_args()
    
    main(args.data_dir, args.checkpoint, args.arch, args.learning_rate, args.hidden_units, args.epochs, args.gpu)
    
