import asyncio
import os
from dotenv import load_dotenv
from lightrag import LightRAG, QueryParam
from lightrag.kg.singlestore_impl import SingleStoreDB
from lightrag.llm.openai import openai_complete, openai_embed
from lightrag.utils import EmbeddingFunc

load_dotenv()
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
WORKING_DIR = os.path.join(ROOT_DIR, "myKG")
if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)
print(f"WorkingDir: {WORKING_DIR}")

# mongo
os.environ["MONGO_URI"] = "mongodb://admin:"+os.environ["S2DB_PASSWORD"]+"@svc-f2513163-96c9-46bb-acd9-360fde26bdb2-mongo.aws-virginia-8.svc.singlestore.com:27017/?authMechanism=PLAIN&tls=true&loadBalanced=true"
os.environ["MONGO_DATABASE"] = "lightrag2"

singlestore_db = SingleStoreDB(
    config={
        "host": "svc-f2513163-96c9-46bb-acd9-360fde26bdb2-dml.aws-virginia-8.svc.singlestore.com",
        "port": 3306,
        "user": "admin",
        "password": os.environ["S2DB_PASSWORD"],
        "database": "lightrag2",
        "workspace": "lightrag-ws",
        "pem_path": "singlestore_bundle.pem",
    }
)
async def main():
    await singlestore_db.initdb()
    await singlestore_db.check_tables()

    rag = LightRAG(
        working_dir=WORKING_DIR,
        llm_model_func=openai_complete,
        llm_model_name="deepseek-ai/deepseek-llm-7b-chat",
        llm_model_max_async=1,
        #llm_model_max_token_size=4096,
        llm_model_max_token_size=4000,
        llm_model_kwargs={"base_url": "https://apps.aws-virginia-nb2.svc.singlestore.com:8000/modelasaservice/da607b11-53f5-4de9-a59f-395724e3b8df/v1", "api_key": os.environ["LLM_API_KEY"]},
        chunk_token_size=500,
        chunk_overlap_token_size=100,
        enable_llm_cache_for_entity_extract=True,
        auto_manage_storages_states=False,
        embedding_func=EmbeddingFunc(
            #embedding_dim=4096,
            embedding_dim=4000,
            #max_token_size=4096,
            max_token_size=4000,
            func=lambda texts: openai_embed(
                texts=texts, model="intfloat/e5-mistral-7b-instruct", base_url="https://apps.aws-virginia-nb2.svc.singlestore.com:8000/modelasaservice/9e207ecc-2a45-411b-8fdd-0ebc638d731d/v1", api_key=os.environ["EMBEDDINGS_API_KEY"]
            ),
        ),
        kv_storage="MongoKVStorage",
        doc_status_storage="MongoDocStatusStorage",
        graph_storage="MongoGraphStorage",
        vector_storage="SingleStoreVectorDBStorage",
    )

    # Set the KV/vector/graph storage's `db` property, so all operation will use same connection pool
    #rag.doc_status.db = singlestore_db
    #rag.full_docs.db = singlestore_db
    #rag.text_chunks.db = singlestore_db
    #rag.llm_response_cache.db = singlestore_db
    #rag.key_string_value_json_storage_cls.db = singlestore_db
    rag.chunks_vdb.db = singlestore_db
    rag.relationships_vdb.db = singlestore_db
    rag.entities_vdb.db = singlestore_db
    #rag.graph_storage_cls.db = singlestore_db
    #rag.chunk_entity_relation_graph.db = singlestore_db
    # add embedding_func for graph database, it's deleted in commit 5661d76860436f7bf5aef2e50d9ee4a59660146c
    rag.chunk_entity_relation_graph.embedding_func = rag.embedding_func

    await rag.initialize_storages()

    file = "./book.txt"
    with open(file, "r", encoding="utf-8") as f:
        await rag.ainsert(f.read())

    print(
        await rag.aquery("What are the top themes in this story?", param=QueryParam(
            mode="hybrid",
            top_k=5,                        # for example
            max_token_for_text_unit=512,    # or 256
            max_token_for_local_context=512,
            max_token_for_global_context=512,
         ))
    )

if __name__ == "__main__":
    asyncio.run(main())
