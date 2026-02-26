import re
import json
import shutil
from pathlib import Path
from typing import List, Dict
from utils import load_config

cfg = load_config()

class ParentStoreManager:
    __store_path: Path

    def __init__(self, store_path = cfg["directories"]["parent_store_path"]):
        self.__store_path = Path(store_path)
        self.__store_path.mkdir(parents=True, exist_ok=True)

    def save(self, parent_id: str, content: str, metadata: Dict):
        parent_file = self.__store_path / f"{parent_id}.json"
        parent_file.write_text(json.dumps({"page_content": content, "metadata": metadata}, 
                                          ensure_ascii=False, indent=2), 
                                encoding="utf-8")

    def save_many(self, parents: List) -> None:
        for parent_id, doc in parents:
            self.save(parent_id, doc.page_content, doc.metadata)

    def load(self, parent_id: str):
        parent_file = self.__store_path / (parent_id if parent_id.lower().endswith(".json") else f"{parent_id}.json")
        return json.loads(parent_file.read_text(encoding="utf-8"))

    def load_content(self, parent_id: str) -> Dict:
        data = self.load(parent_id)
        return {
                "content": data["page_content"],
                "parent_id": parent_id,
                "metadata": data["metadata"]
            }

    @staticmethod
    def _get_sort_key(id_str):
        match = re.search(r'_parent_(\d+)$', id_str)
        return int(match.group(1)) if match else 0

    def load_content_many(self, parent_ids: List[str]) -> List[Dict]:
        unique_ids = set(parent_ids)
        return [self.load_content(pid) for pid in sorted(unique_ids, key=self._get_sort_key)]
    
    def clear_store(self) -> None:
        if self.__store_path.exists():
            shutil.rmtree(self.__store_path)
        self.__store_path.mkdir(parents=True, exist_ok=True)