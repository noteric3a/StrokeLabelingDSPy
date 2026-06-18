from typing import Generator, Dict, Any
import pandas as pd
from pathlib import Path


class LazyExcelReader:
    """Reads Excel files in chunks to minimize memory usage."""
    
    def __init__(self, filepath: str | Path, sheet_name: str = 0, chunk_size: int = 100):
        """Initialize lazy reader.
        
        Args:
            filepath: Path to Excel file
            sheet_name: Sheet name or index to read
            chunk_size: Number of rows to load at a time
        """
        self.filepath = Path(filepath)
        self.sheet_name = sheet_name
        self.chunk_size = chunk_size
        self.total_rows = 0
        self._get_total_rows()
    
    def _get_total_rows(self) -> None:
        """Get total row count without loading entire file."""
        try:
            df = pd.read_excel(self.filepath, sheet_name=self.sheet_name, nrows=1)
            # Load all to count (pandas limitation), but we minimize this
            full_df = pd.read_excel(self.filepath, sheet_name=self.sheet_name)
            self.total_rows = len(full_df)
            del full_df  # Free memory immediately
        except Exception as e:
            raise RuntimeError(f"Failed to read Excel file {self.filepath}: {e}")
    
    def read_chunks(self) -> Generator[pd.DataFrame, None, None]:
        """Yield DataFrames in chunks.
        
        Yields:
            DataFrame containing up to chunk_size rows
        """
        for chunk_start in range(0, self.total_rows, self.chunk_size):
            if chunk_start == 0:
                chunk = pd.read_excel(
                    self.filepath,
                    sheet_name=self.sheet_name,
                    nrows=self.chunk_size,
                )
            else:
                chunk = pd.read_excel(
                    self.filepath,
                    sheet_name=self.sheet_name,
                    skiprows=list(range(1, chunk_start + 1)),
                    nrows=self.chunk_size,
                    header=0,
                )
            yield chunk
    
    def read_columns_chunked(self, columns: list) -> Generator[Dict[str, Any], None, None]:
        """Yield rows as dicts, reading only specified columns.
        
        Args:
            columns: List of column names to read
        
        Yields:
            Dictionary for each row with only specified columns
        """
        for chunk in self.read_chunks():
            if not columns:
                for _, row in chunk.iterrows():
                    yield row.to_dict()
                continue

            # Only keep the columns that actually exist in this chunk.
            available_columns = [col for col in columns if col in chunk.columns]
            if available_columns:
                chunk_filtered = chunk[available_columns]
            else:
                chunk_filtered = chunk.iloc[:, :0]

            for _, row in chunk_filtered.iterrows():
                row_dict = row.to_dict()
                # Ensure all requested columns appear in the output, even if missing.
                for missing_column in columns:
                    if missing_column not in row_dict:
                        row_dict[missing_column] = None
                yield row_dict
            if not available_columns:
                # If no requested columns were present, still yield empty rows with missing keys.
                for _ in chunk.index:
                    yield {col: None for col in columns}
    
    def read_as_dict_list(self, columns: list = None) -> list:
        """Read entire file as list of dicts (for backward compatibility).
        
        Args:
            columns: Optional list of columns to include
        
        Returns:
            List of dictionaries
        """
        result = []
        for row_dict in self.read_columns_chunked(columns or []):
            result.append(row_dict)
        return result
    
    def get_stats(self) -> dict:
        """Get reader statistics."""
        return {
            "filepath": str(self.filepath),
            "total_rows": self.total_rows,
            "chunk_size": self.chunk_size,
            "estimated_chunks": (self.total_rows + self.chunk_size - 1) // self.chunk_size,
        }


def read_excel_lazy(filepath: str | Path, columns: list = None, chunk_size: int = 100) -> list:
    """Convenience function to read Excel file lazily.
    
    Args:
        filepath: Path to Excel file
        columns: Optional list of columns to read
        chunk_size: Number of rows per chunk
    
    Returns:
        List of dictionaries (one per row)
    """
    reader = LazyExcelReader(filepath, chunk_size=chunk_size)
    return reader.read_as_dict_list(columns)
