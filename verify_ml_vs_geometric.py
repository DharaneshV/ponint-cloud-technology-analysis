import os
import glob
import pandas as pd
from run_pipeline import process_single_scan
from analysis.volume import compute_single_volume
import ml_model.infer

def main():
    empty_dir = r"d:\point cloud technology\bgtest-3\empty"
    load_dir = r"d:\point cloud technology\bgtest-3\load"
    
    empty_files = sorted(glob.glob(os.path.join(empty_dir, "SL1*_RD.txt")))
    load_files = sorted(glob.glob(os.path.join(load_dir, "SL1*_RD.txt")))
    
    all_files = empty_files + load_files
    
    results = []
    
    print(f"Verifying {len(all_files)} SL1 scans side-by-side...")
    
    for filepath in all_files:
        filename = os.path.basename(filepath)
        print(f"\nProcessing {filename}...")
        
        # We process the scan once to get the unrotated truck point cloud
        aligned, unrot = process_single_scan(filepath, "scan")
        if unrot is None:
            print(f"  Skipped (extraction failed)")
            continue
            
        # 1. Geometric Pipeline (Scanned Internal Volume)
        vol_m3_geom, _, _, _, _, _ = compute_single_volume(unrot)
        
        # Note: Since we don't have a reference loaded in this loop, we can't run full geometric reference-diff
        # But we know from earlier that geometric classification on these 20 scans was 100% accurate.
        true_label = "Empty" if "empty" in filepath else "Partial"
        
        # 2. Pure ML Pipeline (--ml-only)
        pred_class, conf, pred_vol = ml_model.infer.get_independent_ml_prediction(unrot)
        
        results.append({
            'Filename': filename,
            'True_Class': true_label,
            'ML_Class': pred_class,
            'Class_Match': true_label == pred_class,
            'ML_Confidence': conf,
            'Geom_Vol_m3': vol_m3_geom,
            'ML_Vol_m3': pred_vol
        })
        
        print(f"  True Class: {true_label} | ML Class: {pred_class} (Match: {true_label == pred_class})")
        print(f"  Geom Vol:   {vol_m3_geom:.2f} m3 | ML Vol:   {pred_vol:.2f} m3")
        
    df = pd.DataFrame(results)
    
    acc = df['Class_Match'].mean() * 100
    print("\n--- SIDE-BY-SIDE VERIFICATION RESULTS ---")
    print(f"Total Scans: {len(df)}")
    print(f"Class Agreement Rate (ML vs Ground Truth): {acc:.2f}%")
    
    # Calculate Volume differences
    df['Vol_Diff'] = (df['Geom_Vol_m3'] - df['ML_Vol_m3']).abs()
    print(f"Average Volume Difference (Geom vs ML): {df['Vol_Diff'].mean():.2f} m3")
    
    df.to_csv("side_by_side_results.csv", index=False)
    print("Saved detailed results to side_by_side_results.csv")

if __name__ == "__main__":
    main()
