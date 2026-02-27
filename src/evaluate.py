import numpy as np
import matplotlib.pyplot as plt
import os
import csv

def save_ranked_predictions_csv(predictions_dict, known_genes, new_genes, all_genes, save_path):
    """
    Sorts unlabeled genes by probability and saves to CSV, 
    flagging whether they are true recent discoveries.
    """
    # Isolate the unlabeled pool
    candidate_genes = [g for g in all_genes if g not in known_genes]
    
    # Extract probabilities and sort descending
    candidates_with_probs = [(g, predictions_dict[g]) for g in candidate_genes if g in predictions_dict]
    candidates_with_probs.sort(key=lambda x: x[1], reverse=True)
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Write to CSV
    with open(save_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Rank", "Gene_ID", "Ensemble_Probability", "Is_New_Discovery_Post2022"])
        
        for rank, (gene_id, prob) in enumerate(candidates_with_probs, start=1):
            # Flag as 1 if it's in the 2025 set, 0 otherwise
            is_new = 1 if gene_id in new_genes else 0
            writer.writerow([rank, gene_id, f"{prob:.6f}", is_new])

def calculate_prioritization_metrics(predictions_dict, known_genes, new_genes, all_genes):
    """Evaluates performance and returns metrics + raw ranks for plotting."""
    candidate_genes = [g for g in all_genes if g not in known_genes]
    
    candidates_with_probs = [(g, predictions_dict[g]) for g in candidate_genes if g in predictions_dict]
    candidates_with_probs.sort(key=lambda x: x[1], reverse=True)
    
    sorted_candidate_ids = [x[0] for x in candidates_with_probs]
    total_candidates = len(sorted_candidate_ids)
    
    ranks = []
    valid_new_genes = 0
    
    for gene in new_genes:
        if gene in sorted_candidate_ids:
            valid_new_genes += 1
            ranks.append(sorted_candidate_ids.index(gene) + 1)
            
    if valid_new_genes == 0:
        return None, 0.0, [], total_candidates

    median_rank = np.median(ranks)
    top_1_percent_cutoff = max(1, int(total_candidates * 0.01))
    top_1_percent_genes = set(sorted_candidate_ids[:top_1_percent_cutoff])
    hits_in_top_1 = len(top_1_percent_genes.intersection(new_genes))
    
    fold_enrichment_1_pct = (hits_in_top_1 / valid_new_genes) / 0.01
    
    # Notice we are now returning the ranks and total_candidates as well
    return median_rank, fold_enrichment_1_pct, ranks, total_candidates

def plot_barcode(ranks, total_genes, title, save_path):
    """Generates a barcode plot showing where the validation genes ranked."""
    plt.figure(figsize=(10, 2))
    
    # Draw a vertical line for each rank
    for rank in ranks:
        plt.axvline(x=rank, color='red', alpha=0.6, linewidth=1.5)
        
    plt.xlim(0, total_genes)
    plt.yticks([]) # Hide the y-axis, it's irrelevant for barcodes
    plt.xlabel("Gene Rank (Left = Highest Probability)")
    plt.title(title)
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()