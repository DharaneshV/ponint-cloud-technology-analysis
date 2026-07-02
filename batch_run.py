import os
import glob
import subprocess
import json
import pandas as pd
from datetime import datetime

def main():
    empty_dir = r"d:\point cloud technology\bgtest-3\empty"
    load_dir = r"d:\point cloud technology\bgtest-3\load"
    out_dir = r"d:\point cloud technology\Truck-PointCloud\results\batch_run"
    
    os.makedirs(out_dir, exist_ok=True)
    
    # Get all SL1 empty and load files
    empty_files = sorted(glob.glob(os.path.join(empty_dir, "SL1*_RD.txt")))
    load_files = sorted(glob.glob(os.path.join(load_dir, "SL1*_RD.txt")))
    
    if not empty_files or not load_files:
        print("Could not find SL1 files.")
        return

    # OPTIMIZATION: Use the FIRST empty file as the MASTER REFERENCE
    # This prevents the "bed capacity" from fluctuating slightly between runs due to scan noise.
    master_empty = empty_files[0]
    reference_model = os.path.join(out_dir, "master_ref.pcd")
    
    print(f"--- STEP 1: Building Master Reference Model ---")
    print(f"Using: {os.path.basename(master_empty)}")
    cmd_ref = [
        "python", "cli.py", 
        "--empty", master_empty, 
        "--save-reference", reference_model,
        "--outdir", out_dir
    ]
    subprocess.run(cmd_ref, check=True)
    
    print(f"\n--- STEP 2: Processing All Load Scans ---")
    
    results = []
    
    for idx, load_file in enumerate(load_files):
        filename = os.path.basename(load_file)
        print(f"\nProcessing Load {idx+1}/{len(load_files)}: {filename}")
        
        run_outdir = os.path.join(out_dir, f"run_{idx+1}")
        cmd_load = [
            "python", "cli.py",
            "--reference", reference_model,
            "--load", load_file,
            "--outdir", run_outdir
        ]
        
        try:
            subprocess.run(cmd_load, check=True)
            
            # Read the report
            report_path = os.path.join(run_outdir, "report.json")
            if os.path.exists(report_path):
                with open(report_path, 'r') as f:
                    report = json.load(f)
                    
                vol = report.get('volume', {})
                results.append({
                    "Load_File": filename,
                    "Bed_Capacity_m3": vol.get("bed_capacity_m3"),
                    "Cargo_Volume_m3": vol.get("cargo_volume_m3"),
                    "Remaining_Capacity_m3": vol.get("remaining_capacity_m3"),
                    "Fill_Percentage": vol.get("fill_percentage"),
                })
        except subprocess.CalledProcessError:
            print(f"Failed to process {filename}")
            results.append({
                "Load_File": filename,
                "Error": "Failed"
            })
            
    # Save bulk report
    if results:
        df = pd.DataFrame(results)
        summary_csv = os.path.join(out_dir, "batch_summary.csv")
        df.to_csv(summary_csv, index=False)
        print(f"\n--- BATCH COMPLETE ---")
        print(f"Summary saved to: {summary_csv}")
        print("\nResults Overview:")
        print(df.to_string(index=False))

if __name__ == "__main__":
    main()
