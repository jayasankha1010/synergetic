import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class MultiTaskEmbeddingDataset(Dataset):
    def __init__(self, embeddings_path, labels_dir, all_ids_path):
        """
        Args:
            embeddings_path (str): Path to the txt file with embeddings (Format: ID val1 val2...).
            labels_dir (str): Directory containing txt files for each task/class.
            all_ids_path (str): Path to the txt file containing all valid IDs to process.
        """
        self.embeddings_path = embeddings_path
        self.labels_dir = labels_dir
        self.all_ids_path = all_ids_path
        
        # 1. Load the master list of IDs
        self.ids = self._load_ids(self.all_ids_path)
        
        # 2. Load Embeddings into a dictionary mapped by ID
        self.embeddings_dict = self._load_embeddings()
        
        # 3. Discover tasks (one task per text file in the labels directory)
        # Exclude the master ID file if it happens to be in the same folder
        self.task_files = [f for f in os.listdir(labels_dir) if f.endswith('.txt') and f != os.path.basename(all_ids_path)]
        self.task_files.sort() # Ensure consistent task ordering
        self.num_tasks = len(self.task_files)
        
        # 4. Build the multi-hot label matrix
        self.labels_dict = self._build_label_matrix()
        
        # 5. Filter to only include IDs that have BOTH an embedding and a label entry
        self.valid_ids = [uid for uid in self.ids if uid in self.embeddings_dict]

    def _load_ids(self, filepath):
        with open(filepath, 'r') as f:
            return [line.strip() for line in f if line.strip()]

    def _load_embeddings(self):
        embeddings = {}
        with open(self.embeddings_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                uid = parts[0]
                # Convert the rest of the row to a float32 numpy array
                vector = np.array(parts[1:], dtype=np.float32)
                embeddings[uid] = vector
        return embeddings

    def _build_label_matrix(self):
        # Initialize all labels to 0 for all IDs
        labels = {uid: np.zeros(self.num_tasks, dtype=np.float32) for uid in self.ids}
        
        # Populate the positive labels (1s) based on presence in task files
        for task_idx, task_file in enumerate(self.task_files):
            task_path = os.path.join(self.labels_dir, task_file)
            with open(task_path, 'r') as f:
                positive_ids = set([line.strip() for line in f if line.strip()])
                
            for uid in self.ids:
                if uid in positive_ids:
                    labels[uid][task_idx] = 1.0
                    
        return labels

    def __len__(self):
        return len(self.valid_ids)

    def __getitem__(self, idx):
        uid = self.valid_ids[idx]
        x = torch.tensor(self.embeddings_dict[uid])
        y = torch.tensor(self.labels_dict[uid])
        return x, y

# --- Example Usage ---
if __name__ == "__main__":
    # You will swap these out with your specific paths in your main script
    EMBEDDINGS_FILE = "../data/your_embeddings_file.txt" 
    LABELS_DIRECTORY = "../labels/"
    ALL_IDS_FILE = "../labels/all_genes.txt"
    
    # Instantiate the dataset
    dataset = MultiTaskEmbeddingDataset(
        embeddings_path=EMBEDDINGS_FILE,
        labels_dir=LABELS_DIRECTORY,
        all_ids_path=ALL_IDS_FILE
    )
    
    # Create the DataLoader for batching
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Test the loader
    print(f"Total valid samples: {len(dataset)}")
    print(f"Number of distinct tasks found: {dataset.num_tasks}")
    
    for batch_features, batch_labels in dataloader:
        print(f"Feature batch shape: {batch_features.shape}")
        print(f"Label batch shape: {batch_labels.shape}")
        break # Just printing the first batch to verify