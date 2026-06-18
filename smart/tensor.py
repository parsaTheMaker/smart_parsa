import torch
import time
import sys

# Calculate the number of elements needed for 10 GiB.
# A standard float32 takes 4 bytes of memory.
# 10 GiB = 10 * 1024^3 bytes.
bytes_needed = 20 * 1024**3
num_elements = int(bytes_needed / 4)

device = 'cuda:0'

print(f"Attempting to allocate 10 GiB on {device}...")

try:
    # Create the tensor directly on the target GPU
    tensor = torch.ones(num_elements, dtype=torch.float32, device=device)
    
    print(f"✅ Success! 10 GiB tensor is currently sitting on {device}.")
    print("Press Ctrl+C to release the memory and terminate the script.")
    
    # Infinite loop to keep the script running and the memory allocated
    while True:
        time.sleep(1)

except KeyboardInterrupt:
    print("\nCtrl+C detected. Releasing VRAM...")
    del tensor
    torch.cuda.empty_cache()
    print("Exiting safely.")
    sys.exit(0)
except RuntimeError as e:
    print(f"\n❌ Error: {e}")
    print("Make sure you have at least 5 GPUs (since they are 0-indexed) and enough free VRAM on GPU 4.")
    sys.exit(1)