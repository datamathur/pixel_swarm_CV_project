import torch
import torch.nn.functional as F
from tqdm import tqdm


def train(num_epochs, optimizer, model, device, train_loader, val_loader):
    def closure():
        optimizer.zero_grad()
        outputs = model(inputs.to(device))
        loss = F.cross_entropy(outputs, labels.to(device))
        return loss

    for epoch in tqdm(range(num_epochs)):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            loss = optimizer.step(closure)
            running_loss += loss.item()

            with torch.no_grad():
                outputs = model(inputs.to(device))
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels.to(device)).sum().item()

        print(f'Epoch {epoch + 1}, Train Loss: {running_loss / 100:.3f}, Train Accuracy: {100 * correct / total:.2f}%')

        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = F.cross_entropy(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = 100 * correct / total
        print(f'Validation Loss: {avg_val_loss:.3f}, Validation Accuracy: {val_accuracy:.2f}%')

    print('Finished Training')


def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
    # Fallback to CPU if CUDA is not available
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print(f"Using MPS")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    return device
