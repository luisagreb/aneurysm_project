"""
MASTER SCRIPT: Run All 4 Phases of Thesis Analysis

This script executes the complete analysis pipeline in the correct order:
1. Phase 1: Statistical Discovery (FDR-corrected)
2. Phase 2: Binary Classifier Validation
3. Phase 3: Rescue Experiment
4. Phase 4: 4-Class Stress Test

Author: Antigravity AI
"""

import subprocess
import sys

def run_phase(phase_num, script_name, description):
    """Run a single phase and handle errors."""
    print("\n" + "=" * 80)
    print(f"PHASE {phase_num}: {description}")
    print("=" * 80 + "\n")
    
    try:
        result = subprocess.run(
            [sys.executable, f'src/analysis/{script_name}'],
            check=True,
            capture_output=False
        )
        print(f"\n✅ Phase {phase_num} complete!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Phase {phase_num} failed with error: {e}")
        return False

def main():
    print("=" * 80)
    print("THESIS ANALYSIS PIPELINE - All 4 Phases")
    print("=" * 80)
    
    phases = [
        (1, 'phase1_statistical_discovery.py', 'Statistical Discovery (Healthy vs TAA)'),
        (2, 'phase2_binary_classifier.py', 'Binary Classifier Validation'),
        (3, 'phase3_rescue_analysis.py', 'Rescue Experiment (Collagen Effect)'),
        (4, 'phase4_multiclass_classifier.py', '4-Class Stress Test')
    ]
    
    results = []
    
    for phase_num, script, desc in phases:
        success = run_phase(phase_num, script, desc)
        results.append((phase_num, success))
        
        if not success:
            print(f"\n⚠️  WARNING: Phase {phase_num} failed. Stopping pipeline.")
            print("Fix the error and re-run the master script.")
            break
    
    # Summary
    print("\n" + "=" * 80)
    print("PIPELINE SUMMARY")
    print("=" * 80)
    
    for phase_num, success in results:
        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"Phase {phase_num}: {status}")
    
    if all(success for _, success in results) and len(results) == 4:
        print("\n🎉 ALL PHASES COMPLETE!")
        print("\nResults are saved in:")
        print("  - src/analysis/outputs/phase1/")
        print("  - src/analysis/outputs/phase2/")
        print("  - src/analysis/outputs/phase3/")
        print("  - src/analysis/outputs/phase4/")
    else:
        print("\n⚠️  Pipeline incomplete. Review errors above.")

if __name__ == '__main__':
    main()
