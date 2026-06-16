"""Caching utilities for processed cases to enable resumable runs."""

import json
from pathlib import Path
from typing import Set
import config as cfg


class ProcessingCache:
    """Manages a cache of processed case IDs to skip already-completed work."""
    
    def __init__(self, cache_file: str | Path = cfg.CACHE_FILE):
        """Initialize the cache.
        
        Args:
            cache_file: Path to the cache JSON file
        """
        self.cache_file = Path(cache_file)
        self.processed_cases: Set[str] = self._load_cache()
    
    def _load_cache(self) -> Set[str]:
        """Load cached case IDs from disk."""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    return set(data.get("processed_cases", []))
            except (json.JSONDecodeError, IOError):
                return set()
        return set()
    
    def save_cache(self) -> None:
        """Save current cache state to disk."""
        self.cache_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.cache_file, "w", encoding="utf-8") as f:
            json.dump(
                {"processed_cases": sorted(list(self.processed_cases))},
                f,
                indent=2
            )
    
    def is_processed(self, case_id: str) -> bool:
        """Check if a case has already been processed."""
        return str(case_id) in self.processed_cases
    
    def mark_processed(self, case_id: str) -> None:
        """Mark a case as processed."""
        self.processed_cases.add(str(case_id))
    
    def mark_batch_processed(self, case_ids: list) -> None:
        """Mark multiple cases as processed."""
        for case_id in case_ids:
            self.processed_cases.add(str(case_id))
    
    def clear_cache(self) -> None:
        """Clear all cached entries."""
        self.processed_cases.clear()
        if self.cache_file.exists():
            self.cache_file.unlink()
    
    def get_stats(self) -> dict:
        """Get cache statistics."""
        return {
            "total_cached": len(self.processed_cases),
            "cache_file": str(self.cache_file),
        }
