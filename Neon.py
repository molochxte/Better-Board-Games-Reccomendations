import re
import asyncio
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from typing import Optional, List, Dict, Any

import psycopg2
from psycopg2.extras import RealDictCursor
from sqlalchemy import create_engine, text
from pgvector.sqlalchemy import Vector
from langchain_postgres import PGVector
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from langflow.base.vectorstores.model import LCVectorStoreComponent, check_cached_vector_store
from langflow.base.vectorstores.vector_store_connection_decorator import vector_store_connection
from langflow.helpers.data import docs_to_data
from langflow.inputs.inputs import FloatInput, NestedDictInput
from langflow.io import (
    BoolInput,
    DropdownInput,
    HandleInput,
    IntInput,
    QueryInput,
    SecretStrInput,
    StrInput,
)
from langflow.schema.data import Data
from langflow.serialization import serialize
from langflow.utils.version import get_version_info


@vector_store_connection
class NeonDatabaseVectorStoreComponent(LCVectorStoreComponent):
    display_name: str = "Neon Database"
    description: str = "Ingest and search documents in Neon Database with pgvector"
    documentation: str = "https://neon.tech/docs"
    name = "NeonDatabase"
    icon: str = "PostgreSQL"

    _cached_vector_store: PGVector | None = None

    @dataclass
    class NewDatabaseInput:
        functionality: str = "create"
        fields: dict[str, dict] = field(
            default_factory=lambda: {
                "data": {
                    "node": {
                        "name": "create_database",
                        "description": "Please allow several minutes for creation to complete.",
                        "display_name": "Create new database",
                        "field_order": ["01_new_database_name", "02_branch_name"],
                        "template": {
                            "01_new_database_name": StrInput(
                                name="new_database_name",
                                display_name="Database Name",
                                info="Name of the new database to create in Neon.",
                                required=True,
                            ),
                            "02_branch_name": StrInput(
                                name="branch_name",
                                display_name="Branch Name",
                                info="Branch name for the new database (default: main).",
                                value="main",
                                required=True,
                            ),
                        },
                    },
                }
            }
        )

    @dataclass
    class NewCollectionInput:
        functionality: str = "create"
        fields: dict[str, dict] = field(
            default_factory=lambda: {
                "data": {
                    "node": {
                        "name": "create_collection",
                        "description": "Please allow several seconds for creation to complete.",
                        "display_name": "Create new collection",
                        "field_order": [
                            "01_new_collection_name",
                            "02_dimension",
                            "03_distance_metric",
                        ],
                        "template": {
                            "01_new_collection_name": StrInput(
                                name="new_collection_name",
                                display_name="Collection Name",
                                info="Name of the new collection to create in Neon Database.",
                                required=True,
                            ),
                            "02_dimension": IntInput(
                                name="dimension",
                                display_name="Vector Dimensions",
                                info="Dimensions of the vectors to store.",
                                value=1536,
                                required=True,
                            ),
                            "03_distance_metric": DropdownInput(
                                name="distance_metric",
                                display_name="Distance Metric",
                                info="Distance metric for vector similarity search.",
                                options=["cosine", "euclidean", "inner_product"],
                                value="cosine",
                                required=True,
                            ),
                        },
                    },
                }
            }
        )

    inputs = [
        SecretStrInput(
            name="api_key",
            display_name="Neon API Key",
            info="Authentication API key for accessing Neon Database.",
            value="NEON_API_KEY",
            required=True,
            real_time_refresh=True,
            input_types=[],
        ),
        StrInput(
            name="project_id",
            display_name="Project ID",
            info="The Project ID for the Neon Database instance.",
            required=True,
            real_time_refresh=True,
        ),
        DropdownInput(
            name="database_name",
            display_name="Database",
            info="The Database name for the Neon instance.",
            required=True,
            refresh_button=True,
            real_time_refresh=True,
            dialog_inputs=asdict(NewDatabaseInput()),
            combobox=True,
        ),
        StrInput(
            name="connection_string",
            display_name="Connection String",
            info="Direct connection string to Neon Database. Overrides project_id and database_name.",
            show=False,
        ),
        StrInput(
            name="schema_name",
            display_name="Schema",
            info="Optional schema within Neon Database to use for the collection.",
            advanced=True,
            value="public",
            real_time_refresh=True,
        ),
        DropdownInput(
            name="collection_name",
            display_name="Collection",
            info="The name of the collection within Neon Database where the vectors will be stored.",
            required=True,
            refresh_button=True,
            real_time_refresh=True,
            dialog_inputs=asdict(NewCollectionInput()),
            combobox=True,
            show=False,
        ),
        HandleInput(
            name="embedding_model",
            display_name="Embedding Model",
            input_types=["Embeddings"],
            info="Specify the Embedding Model for vector generation.",
            required=True,
        ),
        *LCVectorStoreComponent.inputs,
        DropdownInput(
            name="search_method",
            display_name="Search Method",
            info="Determine how your content is matched: Vector finds semantic similarity.",
            options=["Vector Search"],
            options_metadata=[{"icon": "SearchVector"}],
            value="Vector Search",
            advanced=True,
            real_time_refresh=True,
        ),
        IntInput(
            name="vector_dimensions",
            display_name="Vector Dimensions",
            info="Dimensions of the embedding vectors.",
            advanced=True,
            value=1536,
        ),
        DropdownInput(
            name="distance_metric",
            display_name="Distance Metric",
            info="Distance metric for vector similarity search.",
            options=["cosine", "euclidean", "inner_product"],
            value="cosine",
            advanced=True,
        ),
        IntInput(
            name="number_of_results",
            display_name="Number of Search Results",
            info="Number of search results to return.",
            advanced=True,
            value=4,
        ),
        DropdownInput(
            name="search_type",
            display_name="Search Type",
            info="Search type to use",
            options=["Similarity", "Similarity with score threshold", "MMR (Max Marginal Relevance)"],
            value="Similarity",
            advanced=True,
        ),
        FloatInput(
            name="search_score_threshold",
            display_name="Search Score Threshold",
            info="Minimum similarity score threshold for search results. "
            "(when using 'Similarity with score threshold')",
            value=0,
            advanced=True,
        ),
        NestedDictInput(
            name="advanced_search_filter",
            display_name="Search Metadata Filter",
            info="Optional dictionary of filters to apply to the search query.",
            advanced=True,
        ),
        BoolInput(
            name="autodetect_collection",
            display_name="Autodetect Collection",
            info="Boolean flag to determine whether to autodetect the collection.",
            advanced=True,
            value=True,
        ),
        StrInput(
            name="content_field",
            display_name="Content Field",
            info="Field to use as the text content field for the vector store.",
            advanced=True,
            value="page_content",
        ),
        StrInput(
            name="metadata_field",
            display_name="Metadata Field",
            info="Field to use for storing document metadata.",
            advanced=True,
            value="metadata",
        ),
        StrInput(
            name="deletion_field",
            display_name="Deletion Based On Field",
            info="When this parameter is provided, documents in the target collection with "
            "metadata field values matching the input metadata field value will be deleted "
            "before new data is loaded.",
            advanced=True,
        ),
        BoolInput(
            name="ignore_invalid_documents",
            display_name="Ignore Invalid Documents",
            info="Boolean flag to determine whether to ignore invalid documents at runtime.",
            advanced=True,
        ),
        NestedDictInput(
            name="neon_vectorstore_kwargs",
            display_name="Neon VectorStore Parameters",
            info="Optional dictionary of additional parameters for the PGVector store.",
            advanced=True,
        ),
    ]

    @classmethod
    async def create_database_api(
        cls,
        new_database_name: str,
        project_id: str,
        api_key: str,
        branch_name: str = "main",
    ):
        """Create a new database in Neon."""
        try:
            import httpx
            
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"https://console.neon.tech/api/v2/projects/{project_id}/databases",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "database": {
                            "name": new_database_name,
                            "branch_id": branch_name,
                        }
                    },
                )
                response.raise_for_status()
                return response.json()
        except Exception as e:
            msg = f"Error creating database: {e}"
            raise ValueError(msg) from e

    @classmethod
    async def create_collection_api(
        cls,
        new_collection_name: str,
        connection_string: str,
        dimension: int,
        distance_metric: str = "cosine",
    ):
        """Create a new collection (table) in Neon Database."""
        try:
            # Create engine
            engine = create_engine(connection_string)
            
            # Create the vector table
            with engine.connect() as conn:
                # Enable pgvector extension
                conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
                
                # Create the collection table
                create_table_sql = f"""
                CREATE TABLE IF NOT EXISTS {new_collection_name} (
                    id SERIAL PRIMARY KEY,
                    {cls._get_content_field()} TEXT NOT NULL,
                    {cls._get_metadata_field()} JSONB,
                    embedding vector({dimension})
                );
                """
                conn.execute(text(create_table_sql))
                
                # Create index based on distance metric
                index_sql = f"""
                CREATE INDEX IF NOT EXISTS {new_collection_name}_embedding_idx 
                ON {new_collection_name} 
                USING ivfflat (embedding vector_{distance_metric}_ops) 
                WITH (lists = 100);
                """
                conn.execute(text(index_sql))
                
                conn.commit()
                
        except Exception as e:
            msg = f"Error creating collection: {e}"
            raise ValueError(msg) from e

    @classmethod
    def _get_content_field(cls) -> str:
        return "page_content"
    
    @classmethod
    def _get_metadata_field(cls) -> str:
        return "metadata"

    @classmethod
    def get_database_list_static(cls, project_id: str, api_key: str):
        """Get list of databases from Neon project."""
        try:
            import httpx
            
            with httpx.Client() as client:
                response = client.get(
                    f"https://console.neon.tech/api/v2/projects/{project_id}/databases",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                )
                response.raise_for_status()
                
                db_list = response.json().get("databases", [])
                db_info_dict = {}
                
                for db in db_list:
                    db_info_dict[db["name"]] = {
                        "id": db["id"],
                        "branch_id": db["branch_id"],
                        "created_at": db["created_at"],
                        "updated_at": db["updated_at"],
                    }
                    
                return db_info_dict
        except Exception as e:
            msg = f"Error fetching database list: {e}"
            raise ValueError(msg) from e

    def get_database_list(self):
        return self.get_database_list_static(
            project_id=self.project_id,
            api_key=self.api_key,
        )

    def get_connection_string(self):
        """Get connection string for Neon Database."""
        if self.connection_string:
            return self.connection_string
            
        if not self.project_id or not self.database_name:
            return None
            
        # Construct connection string from project details
        # This is a simplified version - you may need to fetch actual connection details
        return f"postgresql://user:password@ep-{self.project_id}.us-east-1.aws.neon.tech/{self.database_name}?sslmode=require"

    def get_database_object(self):
        """Get database connection object."""
        try:
            connection_string = self.get_connection_string()
            if not connection_string:
                raise ValueError("No connection string available")
                
            engine = create_engine(connection_string)
            return engine
        except Exception as e:
            msg = f"Error creating database connection: {e}"
            raise ValueError(msg) from e

    def collection_data(self, collection_name: str):
        """Get collection metadata and document count."""
        try:
            engine = self.get_database_object()
            
            with engine.connect() as conn:
                # Check if table exists
                result = conn.execute(text("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_name = :table_name
                    );
                """), {"table_name": collection_name})
                
                if not result.scalar():
                    return None
                
                # Get document count
                count_result = conn.execute(text(f"SELECT COUNT(*) FROM {collection_name}"))
                return count_result.scalar()
                
        except Exception as e:
            self.log(f"Error checking collection data: {e}")
            return None

    def _initialize_database_options(self):
        """Initialize database options for dropdown."""
        try:
            return [
                {
                    "name": name,
                    "id": info["id"],
                    "branch_id": info["branch_id"],
                    "created_at": info["created_at"],
                    "updated_at": info["updated_at"],
                }
                for name, info in self.get_database_list().items()
            ]
        except Exception as e:
            msg = f"Error fetching database options: {e}"
            raise ValueError(msg) from e

    def _initialize_collection_options(self):
        """Initialize collection options for dropdown."""
        try:
            engine = self.get_database_object()
            collections = []
            
            with engine.connect() as conn:
                # Get all tables that have vector columns
                result = conn.execute(text("""
                    SELECT table_name 
                    FROM information_schema.columns 
                    WHERE column_name = 'embedding' 
                    AND table_schema = :schema_name;
                """), {"schema_name": self.schema_name or "public"})
                
                for row in result:
                    table_name = row[0]
                    collections.append({
                        "name": table_name,
                        "records": self.collection_data(table_name),
                        "type": "vector_table",
                    })
                    
            return collections
        except Exception as e:
            self.log(f"Error fetching collection options: {e}")
            return []

    def reset_database_list(self, build_config: dict) -> dict:
        """Reset database list options and related configurations."""
        try:
            database_options = self._initialize_database_options()
            
            database_config = build_config["database_name"]
            database_config.update(
                {
                    "options": [db["name"] for db in database_options],
                    "options_metadata": [{k: v for k, v in db.items() if k != "name"} for db in database_options],
                }
            )
            
            if database_config["value"] not in database_config["options"]:
                database_config["value"] = ""
                build_config["connection_string"]["value"] = ""
                build_config["collection_name"]["show"] = False
                
            database_config["show"] = bool(build_config["api_key"]["value"] and build_config["project_id"]["value"])
            
        except Exception as e:
            self.log(f"Error resetting database list: {e}")
            
        return build_config

    def reset_collection_list(self, build_config: dict) -> dict:
        """Reset collection list options based on provided configuration."""
        try:
            collection_options = self._initialize_collection_options()
            
            collection_config = build_config["collection_name"]
            collection_config.update(
                {
                    "options": [col["name"] for col in collection_options],
                    "options_metadata": [{k: v for k, v in col.items() if k != "name"} for col in collection_options],
                }
            )
            
            if collection_config["value"] not in collection_config["options"]:
                collection_config["value"] = ""
                
            collection_config["show"] = bool(build_config["database_name"]["value"])
            
        except Exception as e:
            self.log(f"Error resetting collection list: {e}")
            
        return build_config

    def reset_build_config(self, build_config: dict) -> dict:
        """Reset all build configuration options to default empty state."""
        database_config = build_config["database_name"]
        database_config.update({"options": [], "options_metadata": [], "value": "", "show": False})
        build_config["connection_string"]["value"] = ""
        
        collection_config = build_config["collection_name"]
        collection_config.update({"options": [], "options_metadata": [], "value": "", "show": False})
        
        # For Neon Database, embedding model is always required
        build_config["embedding_model"]["show"] = True
        build_config["embedding_model"]["required"] = True
        
        return build_config

    async def update_build_config(self, build_config: dict, field_value: str, field_name: str | None = None) -> dict:
        """Update build configuration based on field name and value."""
        if not self.api_key or not self.project_id:
            return self.reset_build_config(build_config)
            
        # Database creation callback
        if field_name == "database_name" and isinstance(field_value, dict):
            if "01_new_database_name" in field_value:
                await self._create_new_database(build_config, field_value)
                return self.reset_collection_list(build_config)
                
        # Collection creation callback
        if field_name == "collection_name" and isinstance(field_value, dict):
            if "01_new_collection_name" in field_value:
                await self._create_new_collection(build_config, field_value)
                return build_config
                
        # Initial execution or token/project change
        first_run = field_name == "collection_name" and not field_value and not build_config["database_name"]["options"]
        if first_run or field_name in {"api_key", "project_id"}:
            return self.reset_database_list(build_config)
            
        # Database selection change
        if field_name == "database_name" and not isinstance(field_value, dict):
            return self._handle_database_selection(build_config, field_value)
            
        # Collection selection change
        if field_name == "collection_name" and not isinstance(field_value, dict):
            return self._handle_collection_selection(build_config, field_value)
            
        return build_config

    async def _create_new_database(self, build_config: dict, field_value: dict) -> None:
        """Create a new database and update build config options."""
        try:
            await self.create_database_api(
                new_database_name=field_value["01_new_database_name"],
                project_id=self.project_id,
                api_key=self.api_key,
                branch_name=field_value.get("02_branch_name", "main"),
            )
        except Exception as e:
            msg = f"Error creating database: {e}"
            raise ValueError(msg) from e
            
        build_config["database_name"]["options"].append(field_value["01_new_database_name"])
        build_config["database_name"]["options_metadata"].append(
            {
                "id": f"db_{field_value['01_new_database_name']}",
                "branch_id": field_value.get("02_branch_name", "main"),
                "created_at": None,
                "updated_at": None,
            }
        )

    async def _create_new_collection(self, build_config: dict, field_value: dict) -> None:
        """Create a new collection and update build config options."""
        try:
            await self.create_collection_api(
                new_collection_name=field_value["01_new_collection_name"],
                connection_string=self.get_connection_string(),
                dimension=field_value.get("02_dimension", 1536),
                distance_metric=field_value.get("03_distance_metric", "cosine"),
            )
        except Exception as e:
            msg = f"Error creating collection: {e}"
            raise ValueError(msg) from e
            
        build_config["collection_name"].update(
            {
                "value": field_value["01_new_collection_name"],
                "options": build_config["collection_name"]["options"] + [field_value["01_new_collection_name"]],
            }
        )
        
        build_config["collection_name"]["options_metadata"].append(
            {
                "records": 0,
                "type": "vector_table",
            }
        )
        
        # Show embedding model input since we're using "Bring your own" approach
        build_config["embedding_model"]["show"] = True
        build_config["embedding_model"]["required"] = True

    def _handle_database_selection(self, build_config: dict, field_value: str) -> dict:
        """Handle database selection and update related configurations."""
        build_config = self.reset_database_list(build_config)
        
        if field_value not in build_config["database_name"]["options"]:
            build_config["database_name"]["value"] = ""
            return build_config
        
        # For Neon Database, embedding model is always required
        build_config["embedding_model"]["show"] = True
        build_config["embedding_model"]["required"] = True
            
        return self.reset_collection_list(build_config)

    def _handle_collection_selection(self, build_config: dict, field_value: str) -> dict:
        """Handle collection selection and update embedding options."""
        build_config["autodetect_collection"]["value"] = True
        build_config = self.reset_collection_list(build_config)
        
        if not field_value:
            return build_config
            
        if field_value and field_value not in build_config["collection_name"]["options"]:
            build_config["collection_name"]["options"].append(field_value)
            build_config["collection_name"]["options_metadata"].append(
                {
                    "records": 0,
                    "type": "vector_table",
                }
            )
            build_config["autodetect_collection"]["value"] = False
        
        # For Neon Database, we always need an embedding model since we don't have built-in vectorize
        build_config["embedding_model"]["show"] = True
        build_config["embedding_model"]["required"] = True
            
        return build_config

    @check_cached_vector_store
    def build_vector_store(self):
        """Build the PGVector store."""
        try:
            from langchain_postgres import PGVector
        except ImportError as e:
            msg = (
                "Could not import langchain_postgres package. "
                "Please install it with `pip install langchain-postgres`."
            )
            raise ImportError(msg) from e

        # Get the embedding model and additional params
        embedding_params = {"embedding": self.embedding_model} if self.embedding_model else {}
        
        if not embedding_params:
            raise ValueError("Embedding model is required for Neon Database vector store")

        # Get the additional parameters
        additional_params = self.neon_vectorstore_kwargs or {}

        # Get Langflow version information
        __version__ = get_version_info()["version"]

        # Get connection details
        connection_string = self.get_connection_string()
        if not connection_string:
            raise ValueError("No connection string available for Neon Database")

        # Bundle up the auto-detect parameters
        autodetect_params = {
            "pre_delete_collection": False,
            "distance_strategy": self._get_distance_strategy(),
        }

        # Attempt to build the Vector Store object
        try:
            vector_store = PGVector(
                connection_string=connection_string,
                embeddings=self.embedding_model,
                collection_name=self.collection_name,
                collection_metadata={"langflow_version": __version__},
                **autodetect_params,
                **additional_params,
            )
        except Exception as e:
            msg = f"Error initializing PGVector: {e}"
            raise ValueError(msg) from e

        # Add documents to the vector store
        self._add_documents_to_vector_store(vector_store)

        return vector_store

    def _get_distance_strategy(self) -> str:
        """Map distance metric to PGVector distance strategy."""
        distance_mapping = {
            "cosine": "COSINE_DISTANCE",
            "euclidean": "EUCLIDEAN_DISTANCE",
            "inner_product": "MAX_INNER_PRODUCT",
        }
        return distance_mapping.get(self.distance_metric, "COSINE_DISTANCE")

    def _add_documents_to_vector_store(self, vector_store) -> None:
        """Add documents to the vector store."""
        self.ingest_data = self._prepare_ingest_data()

        documents = []
        for _input in self.ingest_data or []:
            if isinstance(_input, Data):
                documents.append(_input.to_lc_document())
            else:
                msg = "Vector Store Inputs must be Data objects."
                raise TypeError(msg)

        documents = [
            Document(page_content=doc.page_content, metadata=serialize(doc.metadata, to_str=True)) 
            for doc in documents
        ]

        if documents and self.deletion_field:
            self.log(f"Deleting documents where {self.deletion_field}")
            try:
                # Implement deletion logic if needed
                self.log(f"Deletion field '{self.deletion_field}' specified but deletion not implemented yet.")
            except Exception as e:
                msg = f"Error deleting documents from PGVector based on '{self.deletion_field}': {e}"
                raise ValueError(msg) from e

        if documents:
            self.log(f"Adding {len(documents)} documents to the Vector Store.")
            try:
                vector_store.add_documents(documents)
            except Exception as e:
                msg = f"Error adding documents to PGVector: {e}"
                raise ValueError(msg) from e
        else:
            self.log("No documents to add to the Vector Store.")

    def _map_search_type(self) -> str:
        """Map search type to PGVector search method."""
        search_type_mapping = {
            "Similarity with score threshold": "similarity_score_threshold",
            "MMR (Max Marginal Relevance)": "mmr",
        }
        return search_type_mapping.get(self.search_type, "similarity")

    def _build_search_args(self):
        """Build search arguments for vector store search."""
        query = self.search_query if isinstance(self.search_query, str) and self.search_query.strip() else None

        if not query:
            return {}

        args = {
            "query": query,
            "search_type": self._map_search_type(),
            "k": self.number_of_results,
        }

        if self.search_type == "Similarity with score threshold":
            args["score_threshold"] = self.search_score_threshold

        filter_arg = self.advanced_search_filter or {}
        if filter_arg:
            args["filter"] = filter_arg

        return args

    def search_documents(self, vector_store=None) -> list[Data]:
        """Search documents in the vector store."""
        vector_store = vector_store or self.build_vector_store()

        self.log(f"Search input: {self.search_query}")
        self.log(f"Search type: {self.search_type}")
        self.log(f"Number of results: {self.number_of_results}")

        try:
            search_args = self._build_search_args()
        except Exception as e:
            msg = f"Error in PGVector._build_search_args: {e}"
            raise ValueError(msg) from e

        if not search_args:
            self.log("No search input provided. Skipping search.")
            return []

        docs = []
        search_method = "similarity_search"

        try:
            self.log(f"Calling vector_store.{search_method} with args: {search_args}")
            docs = getattr(vector_store, search_method)(**search_args)
        except Exception as e:
            msg = f"Error performing {search_method} in PGVector: {e}"
            raise ValueError(msg) from e

        self.log(f"Retrieved documents: {len(docs)}")

        data = docs_to_data(docs)
        self.log(f"Converted documents to data: {len(data)}")
        self.status = data

        return data

    def get_retriever_kwargs(self):
        """Get retriever kwargs for the vector store."""
        search_args = self._build_search_args()

        return {
            "search_type": self._map_search_type(),
            "search_kwargs": search_args,
        }
