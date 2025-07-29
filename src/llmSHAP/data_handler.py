from __future__ import annotations
import copy

from llmSHAP.types import Dict, Set, Index, IndexSelection, DataMapping



class DataHandler:
    def __init__(
        self,
        data: DataMapping | str,
        permanent_keys: Set[str] | None = None,
        mask_token: str = "",
    ) -> None:
        self.mask_token: str = mask_token

        if isinstance(data, str):
            data = {index: token for index, token in enumerate(data.split(" "))}

        self.data: DataMapping = dict(data) # Store a *shallow* copy
        self.key_enum: Dict[Index, str] = {index: key for index, key in enumerate(self.data.keys())}

        self.permanent_keys: Set[str] = permanent_keys or set()
        self.permanent_indexes: Set[Index] = {
            index for index, key in self.key_enum.items() if key in self.permanent_keys
        }

    @staticmethod
    def _to_set(selection: IndexSelection) -> Set[Index]:
        """Return a *set* regardless of whether caller passed int or iterable."""
        return {selection} if isinstance(selection, int) else set(selection)

    
    
    def get_feature_enumeration(self) -> Dict[Index, str]:
        return self.key_enum

    def get_keys(self, *, exclude_permanent_keys: bool = False) -> list[Index]:
        """List of indexes, optionally excluding permanent ones."""
        if exclude_permanent_keys:
            return [
                index for index in self.key_enum if index not in self.permanent_indexes
            ]
        return list(self.key_enum.keys())

    def remove(self, indexes: IndexSelection, *, mask: bool = True) -> DataMapping:
        """
        Return a *copy* where the chosen indexes are either masked
        (`mask=True`) or removed (`mask=False`). `self.data` is unchanged.
        """
        index_set = self._to_set(indexes)
        new_data = copy.deepcopy(self.data)

        for index, key in self.key_enum.items():
            if index in index_set:
                if mask: new_data[key] = self.mask_token
                else: new_data.pop(key, None)
        return new_data

    # --------------------------- #
    # !! Destructive removal !!
    # --------------------------- #
    def remove_hard(self, indexes: IndexSelection) -> DataMapping:
        """
        Delete the selected indexes *in-place*. Returns the live mapping.
        """
        index_set = self._to_set(indexes)

        for index in index_set:
            key = self.key_enum.get(index)
            if key is not None:
                self.data.pop(key, None)

        # Re-enumerate because the underlying dict has changed.
        self.key_enum = {index: key for index, key in enumerate(self.data.keys())}
        self.permanent_indexes = {
            index for index, key in self.key_enum.items() if key in self.permanent_keys
        }
        return self.data

    def get_data(
        self,
        indexes: IndexSelection,
        *,
        mask: bool = True,
        exclude_permanent_keys: bool = False,
    ) -> DataMapping:
        """
        Return a dict view according to the supplied options.
        """
        index_set = self._to_set(indexes)

        if not exclude_permanent_keys:
            index_set |= self.permanent_indexes

        return {
            key: (self.data[key] if index in index_set else self.mask_token)
            for index, key in self.key_enum.items()
            if mask or index in index_set
        }

    def to_string(
        self,
        indexes: IndexSelection | None = None,
        *,
        mask: bool = True,
        exclude_permanent_keys: bool = False,
    ) -> str:
        """
        Join the chosen indexes into one space-separated string.
        """
        if indexes is None:
            indexes = self.get_keys(exclude_permanent_keys=exclude_permanent_keys)

        view = self.get_data(indexes, mask=mask, exclude_permanent_keys=exclude_permanent_keys)
        return " ".join(str(value) for value in view.values())