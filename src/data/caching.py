import joblib
import os

# Create a cache directory
CACHE_DIR = '.cache'
os.makedirs(CACHE_DIR, exist_ok=True)

# Configure the memory object
memory = joblib.Memory(CACHE_DIR, verbose=0)

# Expose the decorator
disk_cache = memory.cache