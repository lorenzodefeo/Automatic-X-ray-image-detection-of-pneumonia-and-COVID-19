import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import datasets
import os
import matplotlib.pyplot as plt

# Parametri dell'allenamento
BATCH_SIZE = 32
LEARNING_RATE = 0.0005
EPOCHS = 100
OPTIMIZER = 'Adam'
LOSS_FUNCTION = 'Categorical Cross-Entropy'
H, W = 224, 244

# Transformation for Data Augmentation
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.RandomResizedCrop(size=(224, 224), scale=(0.9, 1.1)),
    transforms.Resize((H, W)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: add_gaussian_noise(x) if torch.rand(1).item() > 0.5 else x),
])

# Dataset e DataLoader
path_to_train = os.path.join(".", "Data", "train")
path_to_test = os.path.join(".", "Data", "test")
train_dataset = datasets.ImageFolder(path_to_train, transform=transform)
test_dataset = datasets.ImageFolder(path_to_test, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)


class LwCoronNet(nn.Module):
    def __init__(self, num_classes=3):
        super(LwCoronNet, self).__init__()

        # Primo blocco convoluzionale
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5, stride=2)
        self.norm1 = nn.BatchNorm2d(num_features=64)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(3, stride=3)

        # Secondo blocco convoluzionale
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3)
        self.norm2 = nn.BatchNorm2d(num_features=128)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(3, stride=3)

        # Terzo blocco convoluzionale
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3)
        self.norm3 = nn.BatchNorm2d(num_features=256)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(3, stride=3)

        # Layer completamente connessi
        self.flatten = nn.Flatten()

        self.norm4 = nn.BatchNorm1d(num_features=2304)
        self.drop4 = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(in_features=2304, out_features=128)  # Inizializzazione per immagini 224x224

        self.relu5 = nn.ReLU()

        # Blocchi aggiuntivi di normalizzazione e dropout
        self.norm5 = nn.BatchNorm1d(num_features=128)
        self.drop5 = nn.Dropout(p=0.5)

        self.fc2 = nn.Linear(in_features=128, out_features=num_classes)

    def forward(self, x):
        # Passaggio attraverso i blocchi convoluzionali
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.norm2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu3(x)
        x = self.pool3(x)

        # Layer completamente connessi
        x = self.flatten(x)
        x = self.norm4(x)
        x = self.drop4(x)
        x = self.fc1(x)

        x = self.relu5(x)

        # Passaggio attraverso i layer aggiuntivi
        x = self.norm5(x)
        x = self.drop5(x)

        x = self.fc2(x)
        return x


# Funzione per aggiungere rumore gaussiano
def add_gaussian_noise(img, mean=0, std=0.25):
    noise = torch.randn_like(img) * std + mean
    noisy_img = img + noise
    noisy_img = torch.clamp(noisy_img, 0., 1.)  # Assicuriamoci che i valori siano tra 0 e 1
    return noisy_img


# Funzione di allenamento
def train_net(model, train_loader, optimizer, criterion, epochs, save_dir='./training_results_undersampling'):
    # Creazione della cartella per salvare i modelli e grafici
    os.makedirs(save_dir, exist_ok=True)

    history = {'loss': [], 'accuracy': []}
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct, total = 0, 0

        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = correct / total
        history['loss'].append(epoch_loss)
        history['accuracy'].append(epoch_acc)

        # Stampa e salvataggio ogni epoca
        message = f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}"
        print(message)

        # Salvataggio del messaggio nel file di log
        log_file_path = os.path.join(save_dir, "training_log.txt")
        save_log_to_file(log_file_path, message)

        # Salvataggio del modello ad ogni epoca
        torch.save(model.state_dict(), os.path.join(save_dir, f"model_epoch_{epoch + 1}.pth"))

        # Salvataggio dei grafici alla fine di ogni epoca
        plot_training_curves(history, save_dir, epoch + 1)

    return history


def save_log_to_file(file_path, message):
    """
    Salva un messaggio in un file di testo. Crea il file se non esiste.

    Args:
        file_path (str): Il percorso del file in cui salvare il messaggio.
        message (str): Il messaggio da salvare.
    """
    # Verifica se il file esiste, altrimenti crealo
    if not os.path.exists(file_path):
        with open(file_path, 'w') as file:  # 'w' crea il file vuoto
            pass  # File creato vuoto, pronto per l'aggiunta di messaggi

    # Aggiunge il messaggio al file
    with open(file_path, 'a') as file:
        file.write(message + '\n')


# Funzione per i grafici
def plot_training_curves(history, save_dir, epoch):
    epochs = range(1, len(history['loss']) + 1)  # Epoche da 1 a N

    # Creazione dei grafici
    plt.figure(figsize=(12, 6))

    # Diagramma della perdita
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['loss'], label='Training Loss', color='red', marker='o')
    plt.title('Loss Diagram')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()

    # Diagramma dell'accuratezza
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['accuracy'], label='Training Accuracy', color='blue', marker='o')
    plt.title('Accuracy Diagram')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()

    # Salvataggio dei grafici
    plt.savefig(os.path.join(save_dir, f"training_curves_epoch_{epoch}.png"))
    plt.close()


if __name__ == '__main__':
    # Definizione dell'ottimizzatore e della loss function
    model = LwCoronNet()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    # Allenamento della rete
    train_net(model, train_loader, optimizer, criterion, epochs=EPOCHS)
