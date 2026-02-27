import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
import csv
from evaluate import calculate_prioritization_metrics, plot_barcode

# Import our custom modules
from data_loader import MultiTaskEmbeddingDataset
from models import SingleDiseaseMLP, CombinedDiseaseMLP
from evaluate import calculate_prioritization_metrics

def get_gene_sets(filepath):
    """Helper to load a set of genes from a text file."""
    if not os.path.exists(filepath):
        return set()
    with open(filepath, 'r') as f:
        return set(line.strip() for line in f if line.strip())

def main():
    # --- Parse Command Line Arguments ---
    parser = argparse.ArgumentParser(description="Train Gene Prioritization Models")
    parser.add_argument("-e", "--exp_name", type=str, required=True, help="Name of the experiment (e.g., 'exp1_baseline')")

    parser.add_argument("--single_dims", type=str, default="256,128,64,64,64,64,64,64,64,32", help="Hidden layers for Single-Task (e.g., '512,256')")
    parser.add_argument("--shared_dims", type=str, default="256,128,64,64,64", help="Shared layers for Multi-Task (e.g., '512,256')")
    parser.add_argument("--head_dims", type=str, default="64,64,64,64,32", help="Task-specific head layers for Multi-Task (e.g., '64,32')")

    args = parser.parse_args()
    experiment_name = args.exp_name

    # Convert comma-separated strings into lists of integers
    single_dims = [int(x) for x in args.single_dims.split(',')]
    shared_dims = [int(x) for x in args.shared_dims.split(',')]
    head_dims = [int(x) for x in args.head_dims.split(',')]

    print(f"Starting Experiment: {experiment_name}")
    print(f"Single-Task Architecture: {single_dims}")
    print(f"Multi-Task Architecture: Shared={shared_dims}, Heads={head_dims}")

    
    print(f"Starting Experiment: {experiment_name}")
    # --- 1. Configuration & Hyperparameters ---
    EMBEDDINGS_FILE = "../data/pubmed_embeddings_rearranged_2022_09.txt"
    LABELS_DIR = "../labels/"
    ALL_IDS_FILE = "../labels/all_genes.txt" # Ensure you have a master list of all genes
    
    INPUT_DIM = 384 # Update this to match your actual embedding dimension
    BATCH_SIZE = 256
    EPOCHS = 50
    LR = 0.001
    L2_LAMBDA = 1e-4 # L2 Regularization parameter
    
    # Device configuration (use GPU if available)
    if torch.cuda.is_available():
        device = torch.device('cuda') # Uses NVIDIA GPUs on the Spartan cluster
    elif torch.backends.mps.is_available():
        device = torch.device('mps')  # Uses Apple Silicon (M-series) GPUs locally
    else:
        device = torch.device('cpu')  # Fallback
    print(f"Training on device: {device}")

    # --- 2. Load Data ---
    dataset = MultiTaskEmbeddingDataset(
        embeddings_path=EMBEDDINGS_FILE,
        labels_dir=LABELS_DIR,
        all_ids_path=ALL_IDS_FILE
    )
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # Identify indices
    cp_idx = dataset.task_files.index("CP-2022-09.txt") if "CP-2022-09.txt" in dataset.task_files else 0
    dee_idx = dataset.task_files.index("DEE_2022_09.txt") if "DEE_2022_09.txt" in dataset.task_files else 1

    # --- NEW: Calculate Positive Weights ---
    # Formula: pos_weight = count_negative_samples / count_positive_samples
    total_samples = len(dataset)
    
    # Calculate for CP
    cp_positives = sum([1 for uid in dataset.valid_ids if dataset.labels_dict[uid][cp_idx] == 1.0])
    cp_negatives = total_samples - cp_positives
    weight_cp = torch.tensor([cp_negatives / cp_positives]).to(device) if cp_positives > 0 else torch.tensor([1.0]).to(device)
    
    # Calculate for DEE
    dee_positives = sum([1 for uid in dataset.valid_ids if dataset.labels_dict[uid][dee_idx] == 1.0])
    dee_negatives = total_samples - dee_positives
    weight_dee = torch.tensor([dee_negatives / dee_positives]).to(device) if dee_positives > 0 else torch.tensor([1.0]).to(device)
    
    print(f"Calculated pos_weight -> CP: {weight_cp.item():.2f}, DEE: {weight_dee.item():.2f}")
    # ---------------------------------------
    
    # Identify which index in the multi-hot label corresponds to which disease
    # (data_loader sorts files alphabetically, so check dataset.task_files)
    cp_idx = dataset.task_files.index("CP-2022-09.txt") if "CP-2022-09.txt" in dataset.task_files else 0
    dee_idx = dataset.task_files.index("DEE_2022_09.txt") if "DEE_2022_09.txt" in dataset.task_files else 1

    # Load Ground Truth sets for Evaluation
    all_genes = set(dataset.valid_ids)
    known_cp = get_gene_sets(os.path.join(LABELS_DIR, "CP-2022-09.txt"))
    known_dee = get_gene_sets(os.path.join(LABELS_DIR, "DEE_2022_09.txt"))
    new_cp = get_gene_sets(os.path.join(LABELS_DIR, "CP-2025_06_vs_2022_09.txt"))
    new_dee = get_gene_sets(os.path.join(LABELS_DIR, "DEE_2025_03_vs_2022_09.txt"))

    # --- 3. Initialize Models & Optimizers ---
    # Single Task Models
    model_cp = SingleDiseaseMLP(input_dim=INPUT_DIM, hidden_dims=single_dims).to(device)
    model_dee = SingleDiseaseMLP(input_dim=INPUT_DIM, hidden_dims=single_dims).to(device)
    
    # Multi-Task Model
    model_combined = CombinedDiseaseMLP(input_dim=INPUT_DIM, shared_dims=shared_dims, head_dims=head_dims).to(device)
    
    # Optimizers (weight_decay applies L2 regularization)
    opt_cp = optim.Adam(model_cp.parameters(), lr=LR, weight_decay=L2_LAMBDA)
    opt_dee = optim.Adam(model_dee.parameters(), lr=LR, weight_decay=L2_LAMBDA)
    opt_combined = optim.Adam(model_combined.parameters(), lr=LR, weight_decay=L2_LAMBDA)
    
    # criterion = nn.BCELoss()
    # We need separate loss functions now because the weights are different
    criterion_cp = nn.BCEWithLogitsLoss(pos_weight=weight_cp)
    criterion_dee = nn.BCEWithLogitsLoss(pos_weight=weight_dee)

    # --- Early Stopping Setup ---
    os.makedirs("../results/checkpoints", exist_ok=True)
    best_loss_mtl = float('inf')
    patience = 8 # If the loss doesn't improve for 8 epochs, stop early
    patience_counter = 0
    best_model_path = f"../results/checkpoints/{experiment_name}_best_mtl.pth"


    # --- 4. Training Loop ---
    for epoch in range(EPOCHS):
        model_cp.train()
        model_dee.train()
        model_combined.train()
        
        total_loss_cp = 0
        total_loss_dee = 0
        total_loss_combined = 0
        
        for features, labels in dataloader:
            features = features.to(device)
            labels = labels.to(device)
            
            # Extract specific labels
            y_cp = labels[:, cp_idx]
            y_dee = labels[:, dee_idx]
            
            # -- Train Single Model (CP) --
            opt_cp.zero_grad()
            pred_cp = model_cp(features)
            loss_cp = criterion_cp(pred_cp, y_cp)
            loss_cp.backward()
            opt_cp.step()
            total_loss_cp += loss_cp.item()
            
            # -- Train Single Model (DEE) --
            opt_dee.zero_grad()
            pred_dee = model_dee(features)
            loss_dee = criterion_dee(pred_dee, y_dee)
            loss_dee.backward()
            opt_dee.step()
            total_loss_dee += loss_dee.item()
            
            # -- Train Combined Model (MTL) --
            opt_combined.zero_grad()
            pred_comb_cp, pred_comb_dee = model_combined(features)
            
            # Joint Loss Calculation: $Loss_{Total} = Loss_{CP} + Loss_{DEE}$
            loss_comb_cp = criterion_cp(pred_comb_cp, y_cp)
            loss_comb_dee = criterion_dee(pred_comb_dee, y_dee)
            loss_combined = loss_comb_cp + loss_comb_dee 
            
            loss_combined.backward()
            opt_combined.step()
            total_loss_combined += loss_combined.item()

        # Print Epoch Summaries
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{EPOCHS}] | "
                  f"Loss CP: {total_loss_cp/len(dataloader):.4f} | "
                  f"Loss DEE: {total_loss_dee/len(dataloader):.4f} | "
                  f"Loss MTL: {total_loss_combined/len(dataloader):.4f}")

        # --- Check Early Stopping for MTL Model ---
        avg_loss_mtl = total_loss_combined / len(dataloader)
        
        if avg_loss_mtl < best_loss_mtl:
            best_loss_mtl = avg_loss_mtl
            patience_counter = 0
            # Save the best weights
            torch.save(model_combined.state_dict(), best_model_path)
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f"\n[!] Early stopping triggered at Epoch {epoch+1}. Model stopped improving.")
            break # Exit the training loop

    # --- 5. Evaluation Phase ---
    print("\n--- Training Complete. Loading Best MTL Model ---")
    
    # Load the best weights back into the model before evaluating
    if os.path.exists(best_model_path):
        model_combined.load_state_dict(torch.load(best_model_path))
    
    model_cp.eval()
    model_dee.eval()
    model_combined.eval()
    
    # Dictionaries to store predictions {gene_id: probability}
    preds_cp_single = {}
    preds_dee_single = {}
    preds_cp_mtl = {}
    preds_dee_mtl = {}
    
    with torch.no_grad():
        for uid in dataset.valid_ids:
            x = torch.tensor(dataset.embeddings_dict[uid]).unsqueeze(0).to(device)
            
            # Apply Sigmoid here to convert logits back to probabilities
            preds_cp_single[uid] = torch.sigmoid(model_cp(x)).item()
            preds_dee_single[uid] = torch.sigmoid(model_dee(x)).item()
            
            p_cp, p_dee = model_combined(x)
            preds_cp_mtl[uid] = torch.sigmoid(p_cp).item()
            preds_dee_mtl[uid] = torch.sigmoid(p_dee).item()

    # --- Calculate Metrics ---
    # 1. CP - Single vs MTL
    # mr_cp_single, fe_cp_single = calculate_prioritization_metrics(preds_cp_single, known_cp, new_cp, all_genes)
    # mr_cp_mtl, fe_cp_mtl = calculate_prioritization_metrics(preds_cp_mtl, known_cp, new_cp, all_genes)
    
    # # 2. DEE - Single vs MTL
    # mr_dee_single, fe_dee_single = calculate_prioritization_metrics(preds_dee_single, known_dee, new_dee, all_genes)
    # mr_dee_mtl, fe_dee_mtl = calculate_prioritization_metrics(preds_dee_mtl, known_dee, new_dee, all_genes)

    # print("\n=== FINAL RESULTS (Post-2022 Discoveries) ===")
    # print(f"CP  (Single Task) -> Median Rank: {mr_cp_single}, Fold Enrich @1%: {fe_cp_single:.2f}")
    # print(f"CP  (Multi-Task)  -> Median Rank: {mr_cp_mtl}, Fold Enrich @1%: {fe_cp_mtl:.2f}")
    # print("------------------------------------------------")
    # print(f"DEE (Single Task) -> Median Rank: {mr_dee_single}, Fold Enrich @1%: {fe_dee_single:.2f}")
    # print(f"DEE (Multi-Task)  -> Median Rank: {mr_dee_mtl}, Fold Enrich @1%: {fe_dee_mtl:.2f}")

    # --- Calculate Metrics & Unpack Ranks ---
    mr_cp_s, fe_cp_s, ranks_cp_s, tot_cp = calculate_prioritization_metrics(preds_cp_single, known_cp, new_cp, all_genes)
    mr_cp_m, fe_cp_m, ranks_cp_m, _ = calculate_prioritization_metrics(preds_cp_mtl, known_cp, new_cp, all_genes)
    
    mr_dee_s, fe_dee_s, ranks_dee_s, tot_dee = calculate_prioritization_metrics(preds_dee_single, known_dee, new_dee, all_genes)
    mr_dee_m, fe_dee_m, ranks_dee_m, _ = calculate_prioritization_metrics(preds_dee_mtl, known_dee, new_dee, all_genes)

    print("\n=== FINAL RESULTS (Post-2022 Discoveries) ===")
    print(f"CP  (Single Task) -> Median Rank: {mr_cp_s}, Fold Enrich @1%: {fe_cp_s:.2f}")
    print(f"CP  (Multi-Task)  -> Median Rank: {mr_cp_m}, Fold Enrich @1%: {fe_cp_m:.2f}")
    print("------------------------------------------------")
    print(f"DEE (Single Task) -> Median Rank: {mr_dee_s}, Fold Enrich @1%: {fe_dee_s:.2f}")
    print(f"DEE (Multi-Task)  -> Median Rank: {mr_dee_m}, Fold Enrich @1%: {fe_dee_m:.2f}")

    # --- Generate Barcode Plots ---
    os.makedirs("../results/plots", exist_ok=True)
    plot_barcode(ranks_cp_s, tot_cp, f"{experiment_name} - CP (Single Task)", f"../results/plots/{experiment_name}_CP_Single.png")
    plot_barcode(ranks_cp_m, tot_cp, f"{experiment_name} - CP (Multi-Task)", f"../results/plots/{experiment_name}_CP_MTL.png")
    plot_barcode(ranks_dee_s, tot_dee, f"{experiment_name} - DEE (Single Task)", f"../results/plots/{experiment_name}_DEE_Single.png")
    plot_barcode(ranks_dee_m, tot_dee, f"{experiment_name} - DEE (Multi-Task)", f"../results/plots/{experiment_name}_DEE_MTL.png")

    # --- Record to Master CSV Ledger ---
    results_file = "../results/experiment_ledger.csv"
    file_exists = os.path.isfile(results_file)
    
    with open(results_file, mode='a', newline='') as f:
        writer = csv.writer(f)
        # Write the header if the file is brand new
        if not file_exists:
            writer.writerow(["Experiment_Name", "Disease", "Architecture", "Median_Rank", "Fold_Enrichment_1%"])
            
        # Append the 4 rows for this experiment
        writer.writerow([experiment_name, "CP", "Single Task", mr_cp_s, fe_cp_s])
        writer.writerow([experiment_name, "CP", "Multi-Task", mr_cp_m, fe_cp_m])
        writer.writerow([experiment_name, "DEE", "Single Task", mr_dee_s, fe_dee_s])
        writer.writerow([experiment_name, "DEE", "Multi-Task", mr_dee_m, fe_dee_m])

    print(f"\nExperiment '{experiment_name}' complete! Results saved to ledger and plots generated.")

if __name__ == "__main__":
    main()