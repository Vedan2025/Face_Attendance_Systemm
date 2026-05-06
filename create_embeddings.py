from MobileFaceNet import MobileFaceNet
import torch
import numpy as np
from torchvision import transforms
from PIL import Image
import os


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MobileFaceNet(embedding_size=512).to(device)
model.load_state_dict(torch.load("model_mobilefacenet.pth", map_location=device))
model.eval()  # Set model to evaluation mode

transform = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

def get_embedding(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = model(image)  # Get 512-D embedding
    return embedding.cpu().numpy()

def load_dataset(dataset_path):
    """Load dataset and compute embeddings."""
    embeddings = {}
    for person in os.listdir(dataset_path):
        person_folder = os.path.join(dataset_path, person)
        if os.path.isdir(person_folder):
            person_embeddings = []
            for img_name in os.listdir(person_folder):
                img_path = os.path.join(person_folder, img_name)     
                try:
                    # image = Image.open(img_path).convert("RGB")
                    embedding = get_embedding(img_path)
                    person_embeddings.append(embedding)
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")
            if person_embeddings:
                embeddings[person] = np.mean(person_embeddings, axis=0)  # Average embedding
    np.save("embeddings.npy", embeddings)
    return embeddings


dataset_path = "dataset"
dataset_embeddings = load_dataset(dataset_path)