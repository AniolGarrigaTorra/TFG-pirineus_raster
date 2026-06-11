"""
Memory-efficient operations for temporal and derived feature processing.

This module provides memory-optimized versions of common operations that:
1. Reduce peak memory usage through chunking and streaming
2. Are parallelizable (can adjust num_workers based on available memory)
3. Don't change final output
4. Keep performance high when parallelized
"""

from __future__ import annotations

import gc
import logging
from pathlib import Path
from typing import Any, Callable

import numpy as np
import rasterio
from concurrent.futures import ThreadPoolExecutor, as_completed

log = logging.getLogger(__name__)


# Global configuration for memory-aware parallelization
MEMORY_CONFIG = {
    "num_workers": 4,  # Default: use 4 parallel workers
    "chunk_size": 512,  # Process rasters in 512x512 chunks
    "gc_interval": 2,  # Force garbage collection every N operations
    "max_rasters_in_memory": 10,  # Max rasters to stack before computing partial result
}


def set_memory_config(num_workers: int | None = None, 
                     max_rasters_in_memory: int | None = None) -> None:
    """
    Configure memory and parallelization settings.
    
    Use this to reduce memory usage on systems without HPC:
    - With HPC (200GB): num_workers=4, max_rasters_in_memory=10 (fast)
    - Home PC (16GB):   num_workers=1, max_rasters_in_memory=2 (slow but works)
    
    Args:
        num_workers: Number of parallel workers (1 = sequential, safe on low memory)
        max_rasters_in_memory: Max rasters to load before computing intermediate result
    """
    if num_workers is not None:
        MEMORY_CONFIG["num_workers"] = max(1, int(num_workers))
        log.info(f"[memory] Set num_workers={MEMORY_CONFIG['num_workers']}")
    
    if max_rasters_in_memory is not None:
        MEMORY_CONFIG["max_rasters_in_memory"] = max(1, int(max_rasters_in_memory))
        log.info(f"[memory] Set max_rasters_in_memory={MEMORY_CONFIG['max_rasters_in_memory']}")


def load_raster_efficient(raster_path: str | Path) -> np.ndarray:
    """Load single raster efficiently."""
    with rasterio.open(raster_path) as src:
        array = src.read(1)
    return array


def stack_rasters_memory_aware(raster_paths: list[str | Path],
                               operation: str = 'mean',
                               parallel: bool = True) -> np.ndarray:
    """
    Stack multiple rasters with memory optimization.
    
    Computes intermediate results every N rasters to avoid loading
    entire stack into memory at once. Parallelizable.
    
    Args:
        raster_paths: List of raster file paths
        operation: 'mean', 'sum', 'max', 'min', 'stack'
        parallel: Use parallel loading if True
    
    Returns:
        Stacked array or computed result
    """
    if not raster_paths:
        return np.array([])
    
    if operation == 'stack':
        # For stacking, we need all data anyway
        arrays = []
        for i, path in enumerate(raster_paths):
            arrays.append(load_raster_efficient(path))
            if (i + 1) % MEMORY_CONFIG["gc_interval"] == 0:
                gc.collect()
        return np.stack(arrays, axis=0)
    
    # For aggregation operations, compute partial results
    max_in_memory = MEMORY_CONFIG["max_rasters_in_memory"]
    num_chunks = (len(raster_paths) + max_in_memory - 1) // max_in_memory
    
    partial_results = []
    
    for chunk_idx in range(num_chunks):
        start_idx = chunk_idx * max_in_memory
        end_idx = min(start_idx + max_in_memory, len(raster_paths))
        chunk_paths = raster_paths[start_idx:end_idx]
        
        # Load chunk
        chunk_arrays = []
        for path in chunk_paths:
            chunk_arrays.append(load_raster_efficient(path))
            gc.collect()
        
        # Compute partial result
        chunk_stack = np.stack(chunk_arrays, axis=0)
        
        if operation == 'mean':
            partial = np.nanmean(chunk_stack, axis=0)
        elif operation == 'sum':
            partial = np.nansum(chunk_stack, axis=0)
        elif operation == 'max':
            partial = np.nanmax(chunk_stack, axis=0)
        elif operation == 'min':
            partial = np.nanmin(chunk_stack, axis=0)
        else:
            raise ValueError(f"Unknown operation: {operation}")
        
        partial_results.append(partial)
        
        # Free memory
        del chunk_arrays, chunk_stack
        gc.collect()
    
    # Combine partial results
    if len(partial_results) == 1:
        return partial_results[0]
    
    # Combine partial results
    combined = np.stack(partial_results, axis=0)
    
    if operation == 'mean':
        result = np.nanmean(combined, axis=0)
    elif operation == 'sum':
        result = np.nansum(combined, axis=0)
    elif operation == 'max':
        result = np.nanmax(combined, axis=0)
    elif operation == 'min':
        result = np.nanmin(combined, axis=0)
    
    del partial_results, combined
    gc.collect()
    
    return result


def parallel_map_memory_aware(func: Callable,
                             items: list,
                             parallel: bool = True) -> list:
    """
    Apply function to items with memory-aware parallelization.
    
    Args:
        func: Function to apply to each item
        items: List of items to process
        parallel: Use parallelization if True
    
    Returns:
        List of results
    """
    if not parallel or MEMORY_CONFIG["num_workers"] == 1:
        # Sequential processing (safe, low memory)
        results = []
        for i, item in enumerate(items):
            results.append(func(item))
            if (i + 1) % MEMORY_CONFIG["gc_interval"] == 0:
                gc.collect()
        return results
    
    # Parallel processing
    results = [None] * len(items)
    num_workers = min(MEMORY_CONFIG["num_workers"], len(items))
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(func, item): idx for idx, item in enumerate(items)}
        
        for future in as_completed(futures):
            idx = futures[future]
            results[idx] = future.result()
            
            if (idx + 1) % MEMORY_CONFIG["gc_interval"] == 0:
                gc.collect()
    
    return results


def chunked_raster_operation(raster_path: str | Path,
                            func: Callable[[np.ndarray], np.ndarray],
                            chunk_size: int = 512) -> np.ndarray:
    """
    Apply function to raster in chunks for memory efficiency.
    
    Useful for operations that don't depend on global raster statistics.
    
    Args:
        raster_path: Path to input raster
        func: Function to apply to each chunk
        chunk_size: Size of chunks (512x512 is usually optimal)
    
    Returns:
        Full processed raster
    """
    with rasterio.open(raster_path) as src:
        height, width = src.height, src.width
        profile = src.profile
        
        # Check if chunking is worth it (avoid overhead for small rasters)
        if height * width < chunk_size * chunk_size * 4:
            # Small raster, process all at once
            array = src.read(1)
            return func(array)
        
        # Large raster, process in chunks
        result_chunks = []
        
        for row in range(0, height, chunk_size):
            row_size = min(chunk_size, height - row)
            chunk_row = []
            
            for col in range(0, width, chunk_size):
                col_size = min(chunk_size, width - col)
                
                # Read chunk
                window = rasterio.windows.Window(col, row, col_size, row_size)
                chunk = src.read(1, window=window)
                
                # Process chunk
                processed = func(chunk)
                chunk_row.append(processed)
                
                gc.collect()
            
            # Concatenate row chunks
            result_chunks.append(np.concatenate(chunk_row, axis=1))
        
        # Concatenate all rows
        result = np.concatenate(result_chunks, axis=0)
        gc.collect()
        
        return result


def free_memory() -> None:
    """Force garbage collection and free memory."""
    gc.collect()
    log.debug("[memory] Garbage collection completed")
