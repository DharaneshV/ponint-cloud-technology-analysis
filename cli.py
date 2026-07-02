import argparse
import sys
import os

# Ensure modules can be imported
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import config
from run_pipeline import process_single_scan, run_pipeline_comparison
from analysis.reference_manager import save_reference_model, load_reference_model

def main():
    parser = argparse.ArgumentParser(description="Truck Point Cloud Processing Pipeline")
    
    # Input options
    parser.add_argument("--empty", type=str, help="Path to raw empty truck point cloud (e.g. .xyz)")
    parser.add_argument("--load", type=str, help="Path to raw loaded truck point cloud (e.g. .xyz)")
    parser.add_argument("--reference", type=str, help="Path to pre-saved empty truck reference model (.pcd)")
    
    # Action options
    parser.add_argument("--save-reference", type=str, help="Path to save the processed empty scan as a reference model (.pcd)")
    
    # Configuration options
    parser.add_argument("--length", type=float, help="Known physical length of the truck in cm", default=None)
    parser.add_argument("--outdir", type=str, help="Directory to save output reports", default="results/runs")
    parser.add_argument("--debugdir", type=str, help="Directory to save intermediate debug PCDs", default=None)
    
    args = parser.parse_args()
    
    if args.length:
        print(f"Setting TRUCK_REFERENCE_LENGTH_CM to {args.length}")
        config.TRUCK_REFERENCE_LENGTH_CM = args.length
        
    debug_dir = args.debugdir if args.debugdir else os.path.join(args.outdir, "debug")
        
    os.makedirs(args.outdir, exist_ok=True)
    os.makedirs(debug_dir, exist_ok=True)
    
    # Mode 1: Save reference only
    if args.empty and args.save_reference and not args.load:
        print("Mode: Generating and saving reference model.")
        empty_aligned, empty_unrot = process_single_scan(args.empty, "empty", debug_dir=debug_dir)
        if empty_unrot is not None:
            save_reference_model(empty_unrot, args.save_reference)
            
            # Compute and report empty bed capacity
            from analysis.volume import compute_empty_volume
            bed_cap_m3, _, _, _, _, _ = compute_empty_volume(empty_unrot)
            
            print(f"\n  +------------------------------------------+")
            print(f"  |  Empty Truck Bed Capacity: {bed_cap_m3:8.4f} m3 |")
            print(f"  +------------------------------------------+\n")
            
            # Save it to report
            import json
            report_file = os.path.join(args.outdir, "empty_report.json")
            with open(report_file, 'w') as f:
                json.dump({"bed_capacity_m3": round(bed_cap_m3, 4)}, f, indent=4)
            print(f"Saved empty volume report to {report_file}")
            
        return

    # For processing a load, we need either --empty or --reference
    if not args.load:
        parser.error("Must provide --load to run volume calculation.")
        
    empty_aligned = None
    empty_unrot = None
    
    # Mode 2: Process from raw empty and load
    if args.empty:
        print("Mode: Processing raw empty and load scans.")
        empty_aligned, empty_unrot = process_single_scan(args.empty, "empty", debug_dir=debug_dir)
        
        # Optionally save it if requested
        if args.save_reference and empty_unrot is not None:
            save_reference_model(empty_unrot, args.save_reference)
            
    # Mode 3: Process using pre-saved reference and raw load
    elif args.reference:
        print("Mode: Using pre-saved reference model.")
        empty_aligned, empty_unrot = load_reference_model(args.reference)
    else:
        parser.error("Must provide either --empty or --reference when processing a load scan.")
        
    if empty_aligned is None or empty_unrot is None:
        print("Failed to obtain empty truck data. Aborting.")
        sys.exit(1)
        
    # Process the load scan
    load_aligned, load_unrot = process_single_scan(args.load, "load", debug_dir=debug_dir)
    
    if load_aligned is None or load_unrot is None:
        print("Failed to process load truck data. Aborting.")
        sys.exit(1)
        
    # Run the comparison pipeline
    run_pipeline_comparison(empty_aligned, empty_unrot, load_aligned, load_unrot, output_dir=args.outdir, debug_dir=debug_dir)

if __name__ == "__main__":
    main()
