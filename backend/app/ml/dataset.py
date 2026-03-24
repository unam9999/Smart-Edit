import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# Define our 14 categories matching our data folders
CATEGORIES = [
    "animal", "architecture", "document", "event", "food", 
    "group", "landscape", "nature", "night", "portrait", 
    "product", "selfie", "vehicle","SUS"
]
# Create a mapping from category name to an integer index (0-13)
CATEGORY_TO_IDX = {cat: idx for idx, cat in enumerate(CATEGORIES)}

class SmartSortDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images (e.g. 'data/')
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []

        # Walk through each category folder to find images
        for cat in CATEGORIES:
            cat_dir = os.path.join(root_dir, cat)
            if not os.path.exists(cat_dir):
                continue
                
            for file_name in os.listdir(cat_dir):
                # Basic check for image extensions
                if file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                    file_path = os.path.join(cat_dir, file_name)
                    self.samples.append((file_path, CATEGORY_TO_IDX[cat]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        # Open image and ensure it has 3 color channels (RGB)
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

def get_dataloaders(data_dir, batch_size=32):
    """
    Creates and returns a DataLoader. 
    (Note: In a full pipeline, this would split data into Train/Val datasets)
    """
    
    # 1. Define Data Augmentation & Preprocessing
    # We resize to 224x224 because that is what EfficientNet-B0 expects
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(), # Data augmentation
        transforms.ColorJitter(brightness=0.2, contrast=0.2), # Data augmentation
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # ImageNet normalization standard
    ])

    # 2. Create the Dataset
    dataset = SmartSortDataset(root_dir=data_dir, transform=transform)

    # 3. Wrap it in a PyTorch DataLoader to handle batching and shuffling
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        # num_workers=2 # Speeds up loading on larger systems
    )
    
    return dataloader, dataset
