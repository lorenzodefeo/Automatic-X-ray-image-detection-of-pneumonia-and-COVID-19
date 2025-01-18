from torchvision import models
from torch import nn
from torchvision.models import ResNet50_Weights


class CustomResNet50(nn.Module):
    def __init__(self, num_classes=3):
        super(CustomResNet50, self).__init__()
        # Carica ResNet-50 pre-addestrato su ImageNet
        self.base_model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)

        # Congela i primi 50 layer (tutti tranne gli ultimi blocchi)
        layers_to_freeze = 50
        for i, child in enumerate(self.base_model.children()):
            if i < layers_to_freeze:
                for param in child.parameters():
                    param.requires_grad = False

        # Modifica della parte finale della rete
        num_features = self.base_model.fc.in_features  # Dimensione dell'output dell'ultimo layer convoluzionale

        # Definisci il nuovo classificatore
        self.base_model.fc = nn.Sequential(
            nn.Linear(num_features, 512),  # Primo Fully Connected
            nn.ReLU(),  # Funzione di attivazione
            nn.Dropout(0.5),  # Dropout per ridurre l'overfitting
            nn.Linear(512, num_classes),  # Strato Fully Connected finale
            nn.Softmax(dim=1)  # Strato softmax per probabilità
        )

    def forward(self, x):
        return self.base_model(x)


if __name__ == '__main__':
    # Istanzia il modello
    classes = 3  # Numero di classi
    model = CustomResNet50(num_classes=classes)

    # Stampa del modello per verificarne la struttura
    print(model)
