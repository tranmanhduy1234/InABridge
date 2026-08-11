"""Manifest-backed vision-language datasets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from PIL import Image, ImageOps
from torch.utils.data import Dataset


def _load_json(path: Path) -> List[Mapping[str, Any]]:
    if path.suffix.lower() != ".json":
        raise ValueError(f"Manifest must be .json or .jsonl: {path}")
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    records = payload if isinstance(payload, list) else payload.get("data")
    if not isinstance(records, list):
        raise ValueError("JSON manifest must be a list or contain a 'data' list")
    return records


class VisionLanguageManifestDataset(Dataset):
    """Lazy RGB image loading with memory-efficient JSONL indexing."""

    def __init__(
        self,
        manifest: str,
        image_root: Optional[str] = None,
        image_column: str = "image",
        text_column: str = "text",
        instruction_column: str = "instruction",
        answer_column: str = "answer",
        stage: int = 2,
        max_samples: Optional[int] = None,
    ) -> None:
        if stage not in (1, 2):
            raise ValueError("stage must be 1 or 2")
        self.manifest = Path(manifest).expanduser().resolve()
        if not self.manifest.is_file():
            raise FileNotFoundError(self.manifest)
        self.records: Optional[List[Mapping[str, Any]]] = None
        self.offsets: Optional[List[int]] = None
        self._jsonl_handle = None
        if self.manifest.suffix.lower() == ".jsonl":
            self.offsets = []
            with self.manifest.open("rb") as handle:
                while max_samples is None or len(self.offsets) < max_samples:
                    offset, line = handle.tell(), handle.readline()
                    if not line:
                        break
                    if line.strip():
                        self.offsets.append(offset)
        else:
            self.records = _load_json(self.manifest)
            if max_samples is not None:
                self.records = self.records[:max_samples]
        if len(self) == 0:
            raise ValueError(f"Manifest contains no samples: {self.manifest}")
        self.image_root = (
            Path(image_root).expanduser().resolve() if image_root else self.manifest.parent
        )
        self.image_column = image_column
        self.text_column = text_column
        self.instruction_column = instruction_column
        self.answer_column = answer_column
        self.stage = stage

    def __len__(self) -> int:
        return len(self.offsets) if self.offsets is not None else len(self.records or ())

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_jsonl_handle"] = None
        return state

    def _record(self, index: int) -> Mapping[str, Any]:
        if self.records is not None:
            return self.records[index]
        assert self.offsets is not None
        if self._jsonl_handle is None:
            self._jsonl_handle = self.manifest.open("rb")
        self._jsonl_handle.seek(self.offsets[index])
        return json.loads(self._jsonl_handle.readline())

    def __getitem__(self, index: int) -> Dict[str, Any]:
        try:
            row = self._record(index)
            image_path = Path(str(row[self.image_column])).expanduser()
            if not image_path.is_absolute():
                image_path = self.image_root / image_path
            with Image.open(image_path) as image:
                loaded = ImageOps.exif_transpose(image).convert("RGB").copy()
            if self.stage == 1:
                return {"image": loaded, "text": str(row[self.text_column])}
            return {
                "image": loaded,
                "instruction": str(row[self.instruction_column]),
                "answer": str(row[self.answer_column]),
            }
        except Exception as error:
            raise RuntimeError(f"Invalid sample {index} in {self.manifest}: {error}") from error
