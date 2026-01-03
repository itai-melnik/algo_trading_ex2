"""
Disk caching for API calls.

Uses joblib for caching expensive API calls to disk.
Handles instance methods properly by ignoring 'self'.
"""
import joblib
import os
from functools import wraps

# Create a cache directory
CACHE_DIR = '.cache'
os.makedirs(CACHE_DIR, exist_ok=True)

# Configure the memory object
memory = joblib.Memory(CACHE_DIR, verbose=0)


def disk_cache(func):
    """
    Decorator for caching function/method results to disk.
    
    Works with both regular functions and instance methods.
    For instance methods, the 'self' argument is ignored in the cache key.
    """
    # Create a cached version that ignores 'self' for methods
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Check if this is a method call (first arg is self)
        if args and hasattr(args[0], func.__name__):
            # This is a method - extract self and remaining args
            self_obj = args[0]
            remaining_args = args[1:]
            
            # Create a cache key from the function name and arguments (excluding self)
            cache_key = (func.__name__,) + remaining_args + tuple(sorted(kwargs.items()))
            
            # Use a simple file-based cache
            import hashlib
            import pickle
            
            key_hash = hashlib.md5(str(cache_key).encode()).hexdigest()
            cache_file = os.path.join(CACHE_DIR, f"{func.__name__}_{key_hash}.pkl")
            
            # Check if cached result exists
            if os.path.exists(cache_file):
                try:
                    with open(cache_file, 'rb') as f:
                        return pickle.load(f)
                except Exception:
                    pass  # Cache read failed, recompute
            
            # Compute result
            result = func(self_obj, *remaining_args, **kwargs)
            
            # Cache the result
            try:
                with open(cache_file, 'wb') as f:
                    pickle.dump(result, f)
            except Exception:
                pass  # Cache write failed, continue anyway
            
            return result
        else:
            # Regular function - use joblib directly
            cached_func = memory.cache(func)
            return cached_func(*args, **kwargs)
    
    return wrapper
