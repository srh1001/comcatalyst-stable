from typing import List
from dataclasses import dataclass

from .var_metadata import VarMetadata

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