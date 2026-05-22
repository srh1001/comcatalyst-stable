from typing import Any
from dataclasses import dataclass

@dataclass
class VarMetadata:
    name     : str
    var_type : str
    support  : Any  # e.g. ["A","B","C"] ou (-1.0, 1.0)

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
