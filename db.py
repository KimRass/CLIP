import sys
sys.path.insert(0, "/Users/jongbeomkim/Desktop/workspace/CLIP")

import torch
import duckdb
import pandas as pd
import numpy as np
from pathlib import Path
import faiss
from pydantic import BaseModel

from utils import load_config, get_device, image_to_grid
from flickr import ImageDataset
from semantic_search import init_faiss_index, get_encoders_from_checkpoint


class Row(BaseModel):
    img_id: int
    img_path: str


class ImageDB:
    def __init__(self, table_name, db_path=None):
        self.table_name = table_name

        if db_path is None:
            self.conn = duckdb.connect(read_only=False)
        else:
            self.conn = duckdb.connect(database=db_path, read_only=False)
        if table_name not in self.conn.execute("SHOW TABLES").df()["name"].tolist():
            self._init_table()
        self.columns = self.conn.table(table_name).columns

    def _drop_table(self):
        self.conn.execute("DROP TABLE IF EXISTS image")

    def _init_table(self):
        self._drop_table()
        self.conn.execute(
            f"""
            CREATE TABLE {self.table_name}(
                img_id INTEGER PRIMARY KEY,
                img_path TEXT
            )
            """
        )

    def show_table(self):
        return image_db.conn.table(self.table_name)

    def _assign_img_id(self, row: Row):
        max_img_id = self.conn.execute(f"SELECT img_id FROM {self.table_name}").df()["img_id"].max()
        row.img_id = int(max_img_id) + 1 if max_img_id.is_integer() else 1
        return row

    def insert(self, row: Row):
        row = self._assign_img_id(row)
        data = pd.DataFrame([row.model_dump()])[self.columns]
        self.conn.append(table_name=self.table_name, df=data)
        return row

    def delete(self, img_id):
        self.conn.execute(f"DELETE FROM {self.table_name} WHERE img_id = {img_id}")

    def _row_to_select_sql(self, row):
        dict_row = row.model_dump()
        sql = f"SELECT * FROM {self.table_name} WHERE "
        for idx, (key, value) in enumerate(dict_row.items()):
            if idx != 0:
                sql += " AND "
            if value is not None:
                if isinstance(value, str):
                    sql += f'{key} = "{value}"'
                else:
                    sql += f"{key} = {value}"
            else:
                sql += f"{key} is NULL"
        return sql

    def select(self, query: Row):
        if query:
            sql = self._row_to_select_sql(query)
            # print(f"[ SQL QUERY STATEMENT: {sql} ]")
            return self.conn.execute(sql).df()
        else:
            return self.conn.execute(f"SELECT * FROM {self.table_name} WHERE img_id = -1").df()

    # def update(self, new_data: Row):
    #     img_id = new_data.img_id
    #     new_data = new_data.model_dump()
    #     del new_data["img_id"]
    #     if new_data:
    #         for key, value in new_data.items():
    #             if isinstance(value, str):
    #                 value = f"'{value}'"
    #             sql = f"UPDATE {self.table_name} SET {key} = {value if value is not None else 'NULL'} WHERE img_id = {img_id}"
    #             self.conn.execute(sql)
    #     df_updated = self.select(Row(img_id=img_id))
    #     return Row(**df_updated.dropna(axis=1).to_dict("records")[0])
image_db = ImageDB(table_name="flickr")
image_db.show_table()


CONFIG = load_config("/Users/jongbeomkim/Desktop/workspace/CLIP/CONFIG.yaml")

DEVICE = get_device()

faiss_idx = init_faiss_index(256)

max_len = 128
ckpt_path = "/Users/jongbeomkim/Documents/clip/checkpoints/clip_flickr.pth"
img_enc, text_enc = get_encoders_from_checkpoint(
    ckpt_path, config=CONFIG, max_len=max_len, device=DEVICE,
)
img_enc.eval()
text_enc.eval()


ds = ImageDataset(data_dir="/Users/jongbeomkim/Documents/datasets/flickr8k_subset", img_size=224)
for idx, (img_path, image) in enumerate(iter(ds), start=1):
    img_embed = img_enc(image[None, :])
    xb = img_embed.detach().cpu().numpy()
    # xb = torch.randn(1, 256).numpy()
    faiss.normalize_L2(xb)
    indices = np.array([idx, ])
    faiss_idx.add_with_ids(xb, indices)

    row = Row(img_id=0, img_path=str(Path(img_path).stem))
    image_db.insert(row)
