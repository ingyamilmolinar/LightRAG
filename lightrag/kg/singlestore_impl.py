import asyncio
import json
import os
import ssl
import sys
import time

import aiomysql
import numpy as np
from dataclasses import dataclass, field
from typing import Any, Dict, List, Union, final

from ..base import BaseVectorStorage
from ..namespace import NameSpace, is_namespace
from ..utils import logger

class SingleStoreDB:
    def __init__(self, config, **kwargs):
        self.pool = None
        self.host = config.get("host", "localhost")
        self.port = config.get("port", 3333)
        self.user = config.get("user", "admin")
        self.password = config.get("password", "password")
        self.database = config.get("database", "database")
        self.workspace = config.get("workspace", "workspace")
        self.pem_path = config.get("pem_path", "singlestore_bundle.pem")
        self.max = 12
        self.increment = 1

        if self.user is None or self.password is None or self.database is None:
            raise ValueError(
                "Missing database user, password, or database in addon_params"
            )

    async def initdb(self):
        # Create SSL context
        ssl_ctx = ssl.create_default_context(cafile=self.pem_path)
        ssl_ctx.check_hostname = False
        ssl_ctx.verify_mode = ssl.CERT_REQUIRED

        # Create connection pool
        self.pool = await aiomysql.create_pool(
            host=self.host,
            port=self.port,
            user=self.user,
            password=self.password,
            db=self.database,
            minsize=1,
            maxsize=self.max,
            autocommit=False,
            ssl=ssl_ctx,
        )
        logger.info(
            f"Connected to SingleStore at {self.host}:{self.port}, database: {self.database}"
        )

    async def close(self):
        if self.pool:
            self.pool.close()
            await self.pool.wait_closed()

    async def check_tables(self):
        """
        Attempt to SELECT 1 from each table. If the table doesn't exist,
        create it using the DDL from TABLES below.
        """
        for table_name, table_info in TABLES.items():
            try:
                await self.query(f"SELECT 1 FROM {table_name} LIMIT 1")
                logger.debug(f"Table '{table_name}' exists.")
            except Exception:
                # Create table if not exist
                try:
                    logger.info(f"Creating table '{table_name}'...")
                    await self.execute(table_info["ddl"])
                except Exception as e:
                    logger.error(f"Error creating table {table_name}: {e}")

    async def query(
        self, sql: str, params: dict = None, multirows: bool = False
    ) -> Union[None, dict, List[dict]]:
        """
        Execute a read-only query; return either None or a row/dict list.
        """
        conn = await self.pool.acquire()
        try:
            cur = await conn.cursor()
            if params:
                await cur.execute(sql, params)
            else:
                await cur.execute(sql)
            rows = await cur.fetchall()
            if not rows:
                return [] if multirows else None
            cols = [desc[0] for desc in cur.description]
            if multirows:
                return [dict(zip(cols, row)) for row in rows]
            return dict(zip(cols, rows[0]))
        finally:
            await cur.close()
            await self.pool.release(conn)

    async def execute(self, sql: str, data: Union[dict, None] = None):
        """
        Execute a write (INSERT, UPDATE, etc.) statement with optional params.
        Commits automatically.
        """
        conn = await self.pool.acquire()
        try:
            cur = await conn.cursor()
            if data:
                await cur.execute(sql, data)
            else:
                await cur.execute(sql)
            await conn.commit()
        finally:
            await cur.close()
            await self.pool.release(conn)

@final
@dataclass
class SingleStoreVectorDBStorage(BaseVectorStorage):
    """
    A single class that can store/query chunk vectors, entity vectors, and
    relationship vectors in SingleStore. Behavior is decided by the `namespace`.
    """
    db: SingleStoreDB = None
    cosine_better_than_threshold: float = float(os.getenv("COSINE_THRESHOLD", "0.2"))
    vector_dimension: int = field(default=4096)

    def __post_init__(self):
        c = self.global_config.get("vector_db_storage_cls_kwargs", {})
        self.cosine_better_than_threshold = c.get(
            "cosine_better_than_threshold", self.cosine_better_than_threshold
        )
        self.vector_dimension = c.get("vector_dimension", self.vector_dimension)
        self._max_batch_size = self.global_config.get("embedding_batch_num", 32)

    async def upsert(self, data: Dict[str, dict]):
        """
        Upsert vectors depending on the namespace:

          - VECTOR_STORE_CHUNKS => upsert into LIGHTRAG_DOC_CHUNKS
          - VECTOR_STORE_ENTITIES => upsert into LIGHTRAG_VDB_ENTITY
          - VECTOR_STORE_RELATIONSHIPS => upsert into LIGHTRAG_VDB_RELATION
        """
        if not data:
            return

        items = list(data.items())
        # Gather items that need embedding
        need_embedding = []
        for _, v in items:
            if "__vector__" not in v and self.embedding_func:
                need_embedding.append(v["content"])

        # If anything needs embedding, embed them in batches
        if need_embedding and self.embedding_func:
            embedded = []
            for i in range(0, len(need_embedding), self._max_batch_size):
                batch = need_embedding[i : i + self._max_batch_size]
                emb = await self.embedding_func(batch)
                embedded.append(emb)
            merged = np.concatenate(embedded)
            idx = 0
            for _, v in items:
                if "__vector__" not in v:
                    v["__vector__"] = merged[idx]
                    idx += 1

        print('Upsert Namespace:', self.namespace)
        # Generate the appropriate SQL depending on namespace
        if is_namespace(self.namespace, NameSpace.VECTOR_STORE_CHUNKS):
            # Upsert doc chunks
            sql = f"""
            INSERT INTO LIGHTRAG_DOC_CHUNKS
                (id, workspace, tokens, chunk_order_index, full_doc_id, content, content_vector)
            VALUES
                (%(id)s, %(workspace)s, %(tokens)s, %(chunk_order_index)s, %(full_doc_id)s, %(content)s,
                 (%(vector_json)s :> VECTOR({self.vector_dimension}))
                )
            ON DUPLICATE KEY UPDATE
                tokens=VALUES(tokens),
                chunk_order_index=VALUES(chunk_order_index),
                full_doc_id=VALUES(full_doc_id),
                content=VALUES(content),
                content_vector=VALUES(content_vector)
            """
            try:
                for k, v in items:
                    print('Item:', k, v)
                    if isinstance(v["__vector__"], np.ndarray):
                        vec_str = json.dumps(v["__vector__"].tolist())
                    else:
                        vec_str = json.dumps(v["__vector__"])
                    p = {
                        "id": k,
                        "workspace": self.db.workspace,
                        "tokens": v.get("tokens", 0),
                        "chunk_order_index": v.get("chunk_order_index", 0),
                        "full_doc_id": v.get("full_doc_id", ""),
                        "content": v.get("content", ""),
                        "vector_json": vec_str,
                    }
                    await self.db.execute(sql, p)
            except Exception as e:
                print('Exception:', e)
                sys.exit(1)

        elif is_namespace(self.namespace, NameSpace.VECTOR_STORE_ENTITIES):
            # Upsert entities
            sql = f"""
            INSERT INTO LIGHTRAG_VDB_ENTITY
                (id, workspace, entity_name, content, content_vector)
            VALUES
                (%(id)s, %(workspace)s, %(entity_name)s, %(content)s,
                 (%(vector_json)s :> VECTOR({self.vector_dimension}))
                )
            ON DUPLICATE KEY UPDATE
                entity_name=VALUES(entity_name),
                content=VALUES(content),
                content_vector=VALUES(content_vector)
            """
            try:
                for k, v in items:
                    print('Item:', k, v)
                    vec_str = (
                        json.dumps(v["__vector__"].tolist())
                        if isinstance(v["__vector__"], np.ndarray)
                        else json.dumps(v["__vector__"])
                    )
                    p = {
                        "id": k,
                        "workspace": self.db.workspace,
                        "entity_name": v.get("entity_name", ""),
                        "content": v.get("content", ""),
                        "vector_json": vec_str,
                    }
                    await self.db.execute(sql, p)
            except Exception as e:
                print('Exception:', e)
                sys.exit(1)

        elif is_namespace(self.namespace, NameSpace.VECTOR_STORE_RELATIONSHIPS):
            # Upsert relationships
            sql = f"""
            INSERT INTO LIGHTRAG_VDB_RELATION
                (id, workspace, source_id, target_id, content, content_vector)
            VALUES
                (%(id)s, %(workspace)s, %(source_id)s, %(target_id)s, %(content)s,
                 (%(vector_json)s :> VECTOR({self.vector_dimension}))
                )
            ON DUPLICATE KEY UPDATE
                source_id=VALUES(source_id),
                target_id=VALUES(target_id),
                content=VALUES(content),
                content_vector=VALUES(content_vector)
            """
            try:
                for k, v in items:
                    print('Item:', k, v)
                    vec_str = (
                        json.dumps(v["__vector__"].tolist())
                        if isinstance(v["__vector__"], np.ndarray)
                        else json.dumps(v["__vector__"])
                    )
                    p = {
                        "id": k,
                        "workspace": self.db.workspace,
                        "source_id": v.get("src_id", ""),
                        "target_id": v.get("tgt_id", ""),
                        "content": v.get("content", ""),
                        "vector_json": vec_str,
                    }
                    await self.db.execute(sql, p)
            except Exception as e:
                print('Exception:', e)
                sys.exit(1)

        else:
            raise ValueError(f"Unsupported namespace: {self.namespace}")

    async def query(self, query: str, top_k: int = 5) -> List[dict[str, Any]]:
        """
        Query vectors depending on the namespace:

         - VECTOR_STORE_CHUNKS => returns [ { "id": "...", "content": "...", "distance": ... }, ...]
         - VECTOR_STORE_ENTITIES => returns [ { "entity_name": "...", "distance": ... }, ...]
         - VECTOR_STORE_RELATIONSHIPS => returns [ { "src_id": "...", "tgt_id": "...", "distance": ... }, ...]
        """
        # Embed the query
        vec = await self.embedding_func([query])
        emb = json.dumps(vec[0].tolist())

        # Decide which table and columns to use
        print('Query Namespace:', self.namespace)
        if is_namespace(self.namespace, NameSpace.VECTOR_STORE_CHUNKS):
            sql = f"""
                SELECT id,
                       content,
                       (content_vector <*> (%(qv)s :> VECTOR({self.vector_dimension}))) AS distance
                FROM LIGHTRAG_DOC_CHUNKS
                WHERE workspace=%(workspace)s
                  AND (content_vector <*> (%(qv)s :> VECTOR({self.vector_dimension}))) > %(threshold)s
                ORDER BY distance DESC
                LIMIT %(top_k)s
            """
            params = {
                "qv": emb,
                "workspace": self.db.workspace,
                "threshold": self.cosine_better_than_threshold,
                "top_k": top_k,
            }
            try:
                rows = await self.db.query(sql, params, multirows=True)
                return rows if rows else []
            except Exception as e:
                print('Exception:', e)
                sys.exit(1)

        elif is_namespace(self.namespace, NameSpace.VECTOR_STORE_ENTITIES):
            sql = f"""
                SELECT entity_name,
                       (content_vector <*> (%(qv)s :> VECTOR({self.vector_dimension}))) AS distance
                FROM LIGHTRAG_VDB_ENTITY
                WHERE workspace=%(workspace)s
                  AND (content_vector <*> (%(qv)s :> VECTOR({self.vector_dimension}))) > %(threshold)s
                ORDER BY distance DESC
                LIMIT %(top_k)s
            """
            params = {
                "qv": emb,
                "workspace": self.db.workspace,
                "threshold": self.cosine_better_than_threshold,
                "top_k": top_k,
            }
            try:
                rows = await self.db.query(sql, params, multirows=True)
                return rows if rows else []
            except Exception as e:
                print('Exception:', e)
                sys.exit(1)

        elif is_namespace(self.namespace, NameSpace.VECTOR_STORE_RELATIONSHIPS):
            sql = f"""
                SELECT source_id AS src_id,
                       target_id AS tgt_id,
                       (content_vector <*> (%(qv)s :> VECTOR({self.vector_dimension}))) AS distance
                FROM LIGHTRAG_VDB_RELATION
                WHERE workspace=%(workspace)s
                  AND (content_vector <*> (%(qv)s :> VECTOR({self.vector_dimension}))) > %(threshold)s
                ORDER BY distance DESC
                LIMIT %(top_k)s
            """
            params = {
                "qv": emb,
                "workspace": self.db.workspace,
                "threshold": self.cosine_better_than_threshold,
                "top_k": top_k,
            }
            try:
                rows = await self.db.query(sql, params, multirows=True)
                return rows if rows else []
            except Exception as e:
                print('Exception:', e)
                sys.exit(1)

        else:
            raise ValueError(f"Unsupported namespace: {self.namespace}")

    async def index_done_callback(self):
        logger.info("Vector data had been saved into SingleStore!")

    async def delete_entity(self, entity_name: str) -> None:
        raise NotImplementedError

    async def delete_entity_relation(self, entity_name: str) -> None:
        raise NotImplementedError


# Update your TABLES dict to include separate tables for entities & relationships
TABLES = {
    # If you only want doc chunks, you can omit the others. 
    # But since you asked about relationships, let's define them all:
    "LIGHTRAG_DOC_CHUNKS": {
        "ddl": f"""
        CREATE TABLE IF NOT EXISTS LIGHTRAG_DOC_CHUNKS (
            id VARCHAR(255),
            workspace VARCHAR(255),
            tokens INT,
            chunk_order_index INT,
            full_doc_id VARCHAR(256),
            content TEXT,
            content_vector VECTOR(4096),
            PRIMARY KEY (workspace, id)
        )
        """.strip()
    },
    "LIGHTRAG_VDB_ENTITY": {
        "ddl": f"""
        CREATE TABLE IF NOT EXISTS LIGHTRAG_VDB_ENTITY (
            id VARCHAR(255),
            workspace VARCHAR(255),
            entity_name VARCHAR(255),
            content TEXT,
            content_vector VECTOR(4096),
            PRIMARY KEY (workspace, id)
        )
        """.strip()
    },
    "LIGHTRAG_VDB_RELATION": {
        "ddl": f"""
        CREATE TABLE IF NOT EXISTS LIGHTRAG_VDB_RELATION (
            id VARCHAR(255),
            workspace VARCHAR(255),
            source_id VARCHAR(255),
            target_id VARCHAR(255),
            content TEXT,
            content_vector VECTOR(4096),
            PRIMARY KEY (workspace, id)
        )
        """.strip()
    },
}

