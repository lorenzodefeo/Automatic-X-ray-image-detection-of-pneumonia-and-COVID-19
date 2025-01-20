import os

import torch
from PIL import Image

from models.LwCoronNet import LwCoronNet

model = LwCoronNet()
dataset_sampling = ''
n_model = max([int(x.replace('model_epoch_', '').replace('.pth', '')) for x in
               os.listdir(os.path.join(".", f"training_results{dataset_sampling}")) if 'pth' in x])
model_path = os.path.join(".", f"training_results{dataset_sampling}", f"model_epoch_{n_model}.pth")
model.load_state_dict(torch.load(model_path, weights_only=True))
model.eval()
DEVICE = "cpu"
CLASSES = ['COVID19', 'NORMAL', 'PNEUMONIA']


def classify_image(image, transform, device=None, class_names=None):
    """
    Funzione per caricare e classificare una singola immagine
    :param image:
    :param model:
    :param transform:
    :param device:
    :param class_names:
    :return:
    """
    if class_names is None:
        class_names = CLASSES.copy()

    if device is None:
        device = DEVICE

    # Applica le trasformazioni
    input_tensor = transform(image).unsqueeze(0)  # Aggiungi una dimensione batch

    # Sposta su dispositivo (GPU o CPU)
    input_tensor = input_tensor.to(device)

    # Disabilita il calcolo dei gradienti per inferenza
    with torch.no_grad():
        # Passa l'immagine attraverso il modello
        output = model(input_tensor)

        # Ottieni la classe con la massima probabilità
        _, predicted_class = torch.max(output, 1)

    # Restituisci il nome della classe predetta
    return class_names[predicted_class.item()], image


def classify_single_image_with_path(image_path, model, transform, device, class_names):
    """
    Funzione per caricare e classificare una singola immagine
    :param image_path:
    :param model:
    :param transform:
    :param device:
    :param class_names:
    :return:
    """
    # Carica l'immagine
    image = Image.open(image_path).convert('RGB')  # Converte in RGB nel caso sia in scala di grigi

    # Applica le trasformazioni
    input_tensor = transform(image).unsqueeze(0)  # Aggiungi una dimensione batch

    # Sposta su dispositivo (GPU o CPU)
    input_tensor = input_tensor.to(device)

    # Disabilita il calcolo dei gradienti per inferenza
    with torch.no_grad():
        # Passa l'immagine attraverso il modello
        output = model(input_tensor)

        # Ottieni la classe con la massima probabilità
        _, predicted_class = torch.max(output, 1)

    # Restituisci il nome della classe predetta
    return class_names[predicted_class.item()]
