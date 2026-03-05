import os
import argparse
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import random

# Import our custom modules
from data_loader import MultiTaskEmbeddingDataset
from models import SingleDiseaseMLP, CombinedDiseaseMLP
from evaluate import calculate_prioritization_metrics, plot_barcode
from evaluate import calculate_prioritization_metrics, plot_barcode, save_ranked_predictions_csv


def set_seed(seed):
    """Ensures each ensemble member starts from a distinct, reproducible state."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def get_gene_sets(filepath):
    if not os.path.exists(filepath):
        return set()
    with open(filepath, 'r') as f:
        return set(line.strip() for line in f if line.strip())

def main():
    # --- Parse Command Line Arguments ---
    parser = argparse.ArgumentParser(description="Ensemble Training for Gene Prioritization")
    parser.add_argument("-e", "--exp_name", type=str, required=True, help="Name of the experiment")
    parser.add_argument("-n", "--num_ensembles", type=int, default=100, help="Number of models to train and average")
    parser.add_argument("--single_dims", type=str, default="512,256,128,64,32", help="Hidden layers for Single-Task")
    parser.add_argument("--shared_dims", type=str, default="512,256,128", help="Shared layers for Multi-Task")
    parser.add_argument("--head_dims", type=str, default="64,32", help="Head layers for Multi-Task")
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout rate")
    args = parser.parse_args()
    
    experiment_name = args.exp_name
    num_ensembles = args.num_ensembles
    single_dims = [int(x) for x in args.single_dims.split(',')]
    shared_dims = [int(x) for x in args.shared_dims.split(',')]
    head_dims = [int(x) for x in args.head_dims.split(',')]
    
    print(f"Starting ENSEMBLE Experiment: {experiment_name} ({num_ensembles} runs)")

    # --- Configuration ---
    EMBEDDINGS_FILE = "../data/pubmed_embeddings_rearranged_2022_09.txt"
    LABELS_DIR = "../labels/"
    ALL_IDS_FILE = "../labels/all_genes.txt"
    INPUT_DIM = 384
    BATCH_SIZE = 256
    EPOCHS = 50
    LR = 0.001
    L2_LAMBDA = 1e-4
    PATIENCE = 8

    # Device configuration
    if torch.cuda.is_available(): device = torch.device('cuda')
    elif torch.backends.mps.is_available(): device = torch.device('mps')
    else: device = torch.device('cpu')

    # --- Load Data & Calculate Weights (Done once for all runs) ---
    dataset = MultiTaskEmbeddingDataset(EMBEDDINGS_FILE, LABELS_DIR, ALL_IDS_FILE)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    cp_idx = dataset.task_files.index("CP-2022-09.txt") if "CP-2022-09.txt" in dataset.task_files else 0
    dee_idx = dataset.task_files.index("DEE_2022_09.txt") if "DEE_2022_09.txt" in dataset.task_files else 1

    all_genes = set(dataset.valid_ids)
    known_cp = get_gene_sets(os.path.join(LABELS_DIR, "CP-2022-09.txt"))
    known_dee = get_gene_sets(os.path.join(LABELS_DIR, "DEE_2022_09.txt"))
    new_cp = get_gene_sets(os.path.join(LABELS_DIR, "CP-2025_06_vs_2022_09.txt"))
    new_dee = get_gene_sets(os.path.join(LABELS_DIR, "DEE_2025_03_vs_2022_09.txt"))

    total_samples = len(dataset)
    cp_positives = sum([1 for uid in dataset.valid_ids if dataset.labels_dict[uid][cp_idx] == 1.0])
    weight_cp = torch.tensor([(total_samples - cp_positives) / cp_positives]).to(device) if cp_positives > 0 else torch.tensor([1.0]).to(device)
    
    dee_positives = sum([1 for uid in dataset.valid_ids if dataset.labels_dict[uid][dee_idx] == 1.0])
    weight_dee = torch.tensor([(total_samples - dee_positives) / dee_positives]).to(device) if dee_positives > 0 else torch.tensor([1.0]).to(device)

    criterion_cp = nn.BCEWithLogitsLoss(pos_weight=weight_cp)
    criterion_dee = nn.BCEWithLogitsLoss(pos_weight=weight_dee)

    # --- Initialize Prediction Accumulators ---
    # We will sum the probabilities here across all runs
    accum_cp_single = {uid: 0.0 for uid in dataset.valid_ids}
    accum_dee_single = {uid: 0.0 for uid in dataset.valid_ids}
    accum_cp_mtl = {uid: 0.0 for uid in dataset.valid_ids}
    accum_dee_mtl = {uid: 0.0 for uid in dataset.valid_ids}

    os.makedirs("../results/checkpoints", exist_ok=True)

    # ==========================================
    # --- THE ENSEMBLE LOOP ---
    # ==========================================
    for run in range(num_ensembles):
        print(f"\n>>> Starting Ensemble Member {run + 1}/{num_ensembles} <<<")
        
        # Set a distinct mathematical seed for this specific run
        set_seed(42 + run) 
        
        # 1. Initialize Fresh Models
        model_cp = SingleDiseaseMLP(INPUT_DIM, single_dims, args.dropout).to(device)
        model_dee = SingleDiseaseMLP(INPUT_DIM, single_dims, args.dropout).to(device)
        model_combined = CombinedDiseaseMLP(INPUT_DIM, shared_dims, head_dims, args.dropout).to(device)
        
        opt_cp = optim.Adam(model_cp.parameters(), lr=LR, weight_decay=L2_LAMBDA)
        opt_dee = optim.Adam(model_dee.parameters(), lr=LR, weight_decay=L2_LAMBDA)
        opt_combined = optim.Adam(model_combined.parameters(), lr=LR, weight_decay=L2_LAMBDA)

        # Early stopping trackers for this specific run
        best_loss_cp, best_loss_dee, best_loss_mtl = float('inf'), float('inf'), float('inf')
        pat_cp, pat_dee, pat_mtl = 0, 0, 0
        
        path_cp = f"../results/checkpoints/{experiment_name}_run{run}_cp.pth"
        path_dee = f"../results/checkpoints/{experiment_name}_run{run}_dee.pth"
        path_mtl = f"../results/checkpoints/{experiment_name}_run{run}_mtl.pth"

        # 2. Train the Models
        for epoch in range(EPOCHS):
            model_cp.train(); model_dee.train(); model_combined.train()
            tot_loss_cp, tot_loss_dee, tot_loss_mtl = 0, 0, 0
            
            for features, labels in dataloader:
                features, labels = features.to(device), labels.to(device)
                y_cp, y_dee = labels[:, cp_idx], labels[:, dee_idx]
                
                # Single CP
                opt_cp.zero_grad()
                loss_cp = criterion_cp(model_cp(features), y_cp)
                loss_cp.backward(); opt_cp.step()
                tot_loss_cp += loss_cp.item()
                
                # Single DEE
                opt_dee.zero_grad()
                loss_dee = criterion_dee(model_dee(features), y_dee)
                loss_dee.backward(); opt_dee.step()
                tot_loss_dee += loss_dee.item()
                
                # Combined MTL
                opt_combined.zero_grad()
                pred_comb_cp, pred_comb_dee = model_combined(features)
                loss_mtl = criterion_cp(pred_comb_cp, y_cp) + criterion_dee(pred_comb_dee, y_dee)
                loss_mtl.backward(); opt_combined.step()
                tot_loss_mtl += loss_mtl.item()

            # Early Stopping Check (Simplified to track MTL for brevity here, 
            # but applies to all if you want strict saving for each)
            avg_mtl = tot_loss_mtl / len(dataloader)
            if avg_mtl < best_loss_mtl:
                best_loss_mtl = avg_mtl; pat_mtl = 0
                torch.save(model_combined.state_dict(), path_mtl)
                torch.save(model_cp.state_dict(), path_cp)   # Saving all when MTL is best
                torch.save(model_dee.state_dict(), path_dee)
            else:
                pat_mtl += 1
                
            if pat_mtl >= PATIENCE:
                print(f"    Early stop triggered at Epoch {epoch+1}")
                break

        # 3. Load Best Weights & Accumulate Predictions
        if os.path.exists(path_cp): model_cp.load_state_dict(torch.load(path_cp))
        if os.path.exists(path_dee): model_dee.load_state_dict(torch.load(path_dee))
        if os.path.exists(path_mtl): model_combined.load_state_dict(torch.load(path_mtl))
        
        model_cp.eval(); model_dee.eval(); model_combined.eval()
        with torch.no_grad():
            for uid in dataset.valid_ids:
                x = torch.tensor(dataset.embeddings_dict[uid]).unsqueeze(0).to(device)
                
                # Add the probability from this run to the running total
                accum_cp_single[uid] += torch.sigmoid(model_cp(x)).item()
                accum_dee_single[uid] += torch.sigmoid(model_dee(x)).item()
                
                p_cp, p_dee = model_combined(x)
                accum_cp_mtl[uid] += torch.sigmoid(p_cp).item()
                accum_dee_mtl[uid] += torch.sigmoid(p_dee).item()

    # ==========================================
    # --- FINAL AVERAGING & EVALUATION ---
    # ==========================================
    print("\n--- All Ensemble Runs Complete. Calculating Final Stable Ranks ---")
    
    # Divide totals by N to get the final averaged probabilities
    final_preds_cp_s = {uid: val / num_ensembles for uid, val in accum_cp_single.items()}
    final_preds_dee_s = {uid: val / num_ensembles for uid, val in accum_dee_single.items()}
    final_preds_cp_m = {uid: val / num_ensembles for uid, val in accum_cp_mtl.items()}
    final_preds_dee_m = {uid: val / num_ensembles for uid, val in accum_dee_mtl.items()}

    # Evaluate
    mr_cp_s, fe_cp_s, ranks_cp_s, tot_cp = calculate_prioritization_metrics(final_preds_cp_s, known_cp, new_cp, all_genes)
    mr_cp_m, fe_cp_m, ranks_cp_m, _ = calculate_prioritization_metrics(final_preds_cp_m, known_cp, new_cp, all_genes)
    
    mr_dee_s, fe_dee_s, ranks_dee_s, tot_dee = calculate_prioritization_metrics(final_preds_dee_s, known_dee, new_dee, all_genes)
    mr_dee_m, fe_dee_m, ranks_dee_m, _ = calculate_prioritization_metrics(final_preds_dee_m, known_dee, new_dee, all_genes)

    # Plot & Save
    os.makedirs("../results/plots", exist_ok=True)
    plot_barcode(ranks_cp_s, tot_cp, f"{experiment_name} - CP (Single Task Ensemble)", f"../results/plots/{experiment_name}_CP_Single_Ensemble.png")
    plot_barcode(ranks_cp_m, tot_cp, f"{experiment_name} - CP (Multi-Task Ensemble)", f"../results/plots/{experiment_name}_CP_MTL_Ensemble.png")
    plot_barcode(ranks_dee_s, tot_dee, f"{experiment_name} - DEE (Single Task Ensemble)", f"../results/plots/{experiment_name}_DEE_Single_Ensemble.png")
    plot_barcode(ranks_dee_m, tot_dee, f"{experiment_name} - DEE (Multi-Task Ensemble)", f"../results/plots/{experiment_name}_DEE_MTL_Ensemble.png")

    results_file = "../results/experiment_ledger.csv"
    file_exists = os.path.isfile(results_file)
    with open(results_file, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Experiment_Name", "Disease", "Architecture", "Median_Rank", "Fold_Enrichment_1%"])
        writer.writerow([experiment_name + "_ensemble", "CP", "Single Task", mr_cp_s, fe_cp_s])
        writer.writerow([experiment_name + "_ensemble", "CP", "Multi-Task", mr_cp_m, fe_cp_m])
        writer.writerow([experiment_name + "_ensemble", "DEE", "Single Task", mr_dee_s, fe_dee_s])
        writer.writerow([experiment_name + "_ensemble", "DEE", "Multi-Task", mr_dee_m, fe_dee_m])

    print("\n=== FINAL ENSEMBLE RESULTS ===")
    print(f"CP  (Single Task) -> Median Rank: {mr_cp_s}, Fold Enrich @1%: {fe_cp_s:.2f}")
    print(f"CP  (Multi-Task)  -> Median Rank: {mr_cp_m}, Fold Enrich @1%: {fe_cp_m:.2f}")
    print("------------------------------------------------")
    print(f"DEE (Single Task) -> Median Rank: {mr_dee_s}, Fold Enrich @1%: {fe_dee_s:.2f}")
    print(f"DEE (Multi-Task)  -> Median Rank: {mr_dee_m}, Fold Enrich @1%: {fe_dee_m:.2f}")

    # --- Export Full Ranked Lists to CSV ---
    print("\n--- Exporting Ranked Gene Lists to CSV ---")
    rankings_dir = f"../results/rankings/{experiment_name}"
    os.makedirs(rankings_dir, exist_ok=True)
    
    save_ranked_predictions_csv(final_preds_cp_s, known_cp, new_cp, all_genes, 
                                f"{rankings_dir}/CP_Single_Ranks.csv")
    
    save_ranked_predictions_csv(final_preds_cp_m, known_cp, new_cp, all_genes, 
                                f"{rankings_dir}/CP_MTL_Ranks.csv")
    
    save_ranked_predictions_csv(final_preds_dee_s, known_dee, new_dee, all_genes, 
                                f"{rankings_dir}/DEE_Single_Ranks.csv")
    
    save_ranked_predictions_csv(final_preds_dee_m, known_dee, new_dee, all_genes, 
                                f"{rankings_dir}/DEE_MTL_Ranks.csv")
    
    print(f"Rankings saved successfully to {rankings_dir}/")

if __name__ == "__main__":
    main()