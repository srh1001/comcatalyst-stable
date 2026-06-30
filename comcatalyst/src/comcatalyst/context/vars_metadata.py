from typing import List, Any
from dataclasses import dataclass
import numpy as np

@dataclass
class VarMetadata:
    name: str
    var_type: str
    support: Any  # e.g. ["A","B","C"] ou (-1.0, 1.0)

    _SUPPORTED_TYPES = {"categorical", "continuous"}

    def __post_init__(self):
        if self.var_type not in self._SUPPORTED_TYPES:
            raise ValueError(
                f"Variable type '{self.var_type}' not supported."
                f"Use supported types: {self._SUPPORTED_TYPES}"
            )
        
    # types considérés comme catégoriels au sens large
    _CATEG_LIKE_TYPES = {"categorical"}

    def is_categ_like(self) -> bool:
        return self.var_type in self._CATEG_LIKE_TYPES

    def is_categorical(self) -> bool:
        return self.var_type == "categorical"

    def is_continuous(self) -> bool:
        return self.var_type == "continuous"

@dataclass
class VarsMetadata:
    vars_metadata: List[VarMetadata]

    def __getitem__(self, key: int) -> VarMetadata:
        return self.vars_metadata[key]
    
    def __iter__(self):
        return iter(self.vars_metadata)
    
    def __len__(self):
        return len(self.vars_metadata)
    
    def get_categ_like_vars_indexes(self) -> List[int]:
        return [idx for idx, m in enumerate(self.vars_metadata) if m.is_categ_like()]
    
    def get_categorical_vars_indexes(self) -> List[int]:
        return [idx for idx, m in enumerate(self.vars_metadata) if m.is_categorical()]
        
    def get_continuous_vars_indexes(self) -> List[int]:
        return [idx for idx, m in enumerate(self.vars_metadata) if m.is_continuous()]
    
    def get_ordinal_vars_indexes(self) -> List[int]:
        return [idx for idx, m in enumerate(self.vars_metadata) if m.is_ordinal()]
    
    def get_categ_like_vars_metadata(self) -> List[VarMetadata]:
        return [m for m in self.vars_metadata if m.is_categ_like()]
    
    def get_categorical_vars_metadata(self) -> List[VarMetadata]:
        return [m for m in self.vars_metadata if m.is_categorical()]
        
    def get_continuous_vars_metadata(self) -> List[VarMetadata]:
        return [m for m in self.vars_metadata if m.is_continuous()]
    
    def get_ordinal_vars_metadata(self) -> List[VarMetadata]:
        return [m for m in self.vars_metadata if m.is_ordinal()]
    
    def str_values(self, values: np.ndarray) -> str:
        parts = []
        for val, m in zip(values, self.vars_metadata):
            if m.is_categ_like():
                label = m.support[int(val)]
                parts.append(f"{m.name} = {label}")
            else:
                parts.append(f"{m.name} = {val:.4f}")
        return " / ".join(parts)