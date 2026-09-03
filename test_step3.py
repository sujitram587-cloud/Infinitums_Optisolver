from mps_parser import MPSParser

def main():
    print("=== Testing Step 3: Sovereign MPS File Parser ===")
    
    mps_file = "sample_model.mps"
    parser = MPSParser(mps_file)
    
    model, name = parser.parse()
    print(f"[SUCCESS] Parsed MPS Model: '{name}'")
    print(f"  -> Variables: {model.num_vars}")
    print(f"  -> Constraints: {model.num_constraints}")
    print(f"  -> Non-zero coefficients (NNZ): {model.nnz}")

    # Offload parsed MPS model straight into RTX 4060 VRAM
    success = model.offload_to_gpu()
    if success:
        print("[SUCCESS] Parsed MPS dataset offloaded into RTX 4060 VRAM successfully!")
    else:
        print("[ERROR] Failed to offload MPS model to GPU.")

    # Cleanup memory
    model.free()
    print("[SUCCESS] VRAM and CPU Memory cleared successfully.")

if __name__ == "__main__":
    main()