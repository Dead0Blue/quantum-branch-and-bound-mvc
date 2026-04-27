import os
import sys
import networkx as nx

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from classical_solvers import brute_force_mvc, bb_mvc_baseline, bb_mvc_improved
from montanaro_emulation import Montanaro_inspired_raw, Montanaro_inspired_enhanced
from instance_generator import generate_mvc_instance

def run_sanity_checks():
    print("==========================================================")
    print("Sanity Checks: Verifying exactness of classical heuristics")
    print("==========================================================")
    
    # Generate a few small graphs
    instances = []
    for seed in [10, 42, 100]:
        instances.append(generate_mvc_instance(model="erdos_renyi", n=8, p=0.4, seed=seed))
        instances.append(generate_mvc_instance(model="barabasi_albert", n=10, m=2, seed=seed))
        instances.append(generate_mvc_instance(model="regular", n=12, d=3, seed=seed))
        
    all_passed = True
    
    print(f"{'Graph Type':<15} | {'n':<3} | {'Brute':<5} | {'Base B&B':<8} | {'Imp B&B':<7} | {'Raw Mont':<8} | {'Enh Mont':<8}")
    print("-" * 75)
    
    for inst in instances:
        G = inst['graph']
        n = G.number_of_nodes()
        family = inst.get('model', 'unknown')[:10]
        
        # 1. Brute force
        bf_cover, bf_size = brute_force_mvc(G)
        
        # 2. Baseline B&B
        res_base = bb_mvc_baseline(G)
        assert res_base['cover_size'] == bf_size, f"Baseline failed on {family} n={n}"
        
        # 3. Improved B&B
        res_imp = bb_mvc_improved(G)
        assert res_imp['cover_size'] == bf_size, f"Improved failed on {family} n={n}"
        
        # 4. Raw Emulation
        res_raw = Montanaro_inspired_raw(G)
        if res_raw['cover_size'] != bf_size:
            print(f"Raw Emulation failed on {family} n={n}: expected {bf_size}, got {res_raw['cover_size']}")
            all_passed = False
        
        # 5. Enhanced Emulation
        res_enh = Montanaro_inspired_enhanced(G)
        if res_enh['cover_size'] != bf_size:
            print(f"Enhanced Emulation failed on {family} n={n}: expected {bf_size}, got {res_enh['cover_size']}")
            print(f"  Enhanced Emulation returned state: {res_enh}")
            all_passed = False
            
        print(f"{family:<15} | {n:<3} | {bf_size:<5} | {res_base['cover_size']:<8} | {res_imp['cover_size']:<7} | {res_raw['cover_size']:<8} | {res_enh['cover_size']:<8}")
        
    print("-" * 75)
    
    if all_passed:
        print("\nAll sanity checks passed! Exactness is preserved.")
        print("The enhanced rules (preprocessing, forced neighbors, maximal matching lower bound)")
        print("are mathematically sound exact reductions.")
    else:
        print("\nSanity checks failed! Check the assertions.")

if __name__ == "__main__":
    run_sanity_checks()
