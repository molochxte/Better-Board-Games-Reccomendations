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
    description: str = "Ingest and search documents in Neon Database with pgvector. Handles pre-chunked data."
    documentation: str = "https://neon.tech/docs"
    name = "NeonDatabase"
    icon: str = "PostgreSQL"

    _cached_vector_store: PGVector | None = None

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
        StrInput(
            name="connection_string",
            display_name="Connection String",
            info="Direct connection string to your Neon Database. Get this from Neon Console → Connection Details. Format: postgresql://user:pass@host:port/db?sslmode=require",
            required=True,
            real_time_refresh=True,
        ),
        DropdownInput(
            name="collection_name",
            display_name="Collection",
            info="The name of the collection (table) within Neon Database where the vectors will be stored.",
            required=True,
            refresh_button=True,
            real_time_refresh=True,
            dialog_inputs=asdict(NewCollectionInput()),
            combobox=True,
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
        IntInput(
            name="batch_size",
            display_name="Batch Size",
            info="Number of documents to process in each batch to avoid token limits.",
            advanced=True,
            value=10,
        ),
        StrInput(
            name="output_template",
            display_name="Output Template",
            info="Template for formatting BoardGameGeek search results. Available placeholders: {id}, {name}, {game_name}, {description}, {domains}, {mechanics}, {year_published}, {min_players}, {max_players}, {play_time}, {min_age}, {users_rated}, {rating_average}, {bgg_rank}, {complexity_average}, {owned_users}, {url}, {rank}, {score}. Leave empty for JSON output. Use 'DEBUG' to see available fields.",
            advanced=True,
            value="Game: {name}\nYear: {year_published}\nRating: {rating_average}/10 (BGG Rank #{bgg_rank})\nPlayers: {min_players}-{max_players} | Time: {play_time}min | Age: {min_age}+\nType: {domains}\nMechanics: {mechanics}\nComplexity: {complexity_average}/5\nDescription: {description}",
        ),
        NestedDictInput(
            name="neon_vectorstore_kwargs",
            display_name="Neon VectorStore Parameters",
            info="Optional dictionary of additional parameters for the PGVector store.",
            advanced=True,
        ),
    ]

    @classmethod
    def _get_content_field(cls) -> str:
        return "page_content"
    
    @classmethod
    def _get_metadata_field(cls) -> str:
        return "metadata"

    def get_database_object(self):
        """Get database connection object."""
        try:
            if not self.connection_string:
                raise ValueError("Connection string is required")
                
            engine = create_engine(self.connection_string)
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
                    AND table_schema = 'public';
                """))
                
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

    def reset_collection_list(self, build_config: dict) -> dict:
        """Reset collection list options based on provided configuration."""
        try:
            if not self.connection_string:
                collection_config = build_config["collection_name"]
                collection_config.update({"options": [], "options_metadata": [], "value": "", "show": False})
                return build_config
                
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
                
            collection_config["show"] = bool(self.connection_string)
            
        except Exception as e:
            self.log(f"Error resetting collection list: {e}")
            
        return build_config

    def reset_build_config(self, build_config: dict) -> dict:
        """Reset all build configuration options to default empty state."""
        collection_config = build_config["collection_name"]
        collection_config.update({"options": [], "options_metadata": [], "value": "", "show": False})
        
        # For Neon Database, embedding model is always required
        build_config["embedding_model"]["show"] = True
        build_config["embedding_model"]["required"] = True
        
        return build_config

    async def update_build_config(self, build_config: dict, field_value: str, field_name: str | None = None) -> dict:
        """Update build configuration based on field name and value."""
        # Collection creation callback
        if field_name == "collection_name" and isinstance(field_value, dict):
            if "01_new_collection_name" in field_value:
                await self._create_new_collection(build_config, field_value)
                return build_config
                
        # Connection string change
        if field_name == "connection_string":
            return self.reset_collection_list(build_config)
            
        # Collection selection change
        if field_name == "collection_name" and not isinstance(field_value, dict):
            return self._handle_collection_selection(build_config, field_value)
            
        # Initial execution
        first_run = field_name == "collection_name" and not field_value and not build_config["collection_name"]["options"]
        if first_run:
            return self.reset_collection_list(build_config)
            
        return build_config

    async def _create_new_collection(self, build_config: dict, field_value: dict) -> None:
        """Create a new collection and update build config options."""
        try:
            await self.create_collection_api(
                new_collection_name=field_value["01_new_collection_name"],
                connection_string=self.connection_string,
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
        if not self.connection_string:
            raise ValueError("Connection string is required for Neon Database")

        # Bundle up the auto-detect parameters
        autodetect_params = {
            "pre_delete_collection": False,
            "distance_strategy": self._get_distance_strategy(),
            "create_extension": True,
        }

        # Attempt to build the Vector Store object
        try:
            vector_store = PGVector(
                connection=self.connection_string,
                embeddings=self.embedding_model,
                collection_name=self.collection_name,
                collection_metadata={"langflow_version": __version__},
                **autodetect_params,
                **additional_params,
            )
        except Exception as e:
            msg = f"Error initializing PGVector: {e}"
            raise ValueError(msg) from e

        # Add documents to the vector store in batches
        self._add_documents_to_vector_store_batched(vector_store)

        return vector_store

    def _get_distance_strategy(self) -> str:
        """Map distance metric to PGVector distance strategy."""
        distance_mapping = {
            "cosine": "cosine",
            "euclidean": "euclidean", 
            "inner_product": "inner_product",
        }
        return distance_mapping.get(self.distance_metric, "cosine")

    def _add_documents_to_vector_store_batched(self, vector_store) -> None:
        """Add documents to the vector store in batches to avoid token limits."""
        self.ingest_data = self._prepare_ingest_data()

        if not self.ingest_data:
            self.log("No documents to add to the Vector Store.")
            return

        documents = []
        for _input in self.ingest_data:
            if isinstance(_input, Data):
                documents.append(_input.to_lc_document())
            else:
                msg = "Vector Store Inputs must be Data objects."
                raise TypeError(msg)

        # Convert to Document objects with serialized metadata
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
            self.log(f"Adding {len(documents)} documents to the Vector Store in batches of {self.batch_size}.")
            
            # Process documents in batches
            total_batches = (len(documents) + self.batch_size - 1) // self.batch_size
            
            for i in range(0, len(documents), self.batch_size):
                batch = documents[i:i + self.batch_size]
                batch_num = (i // self.batch_size) + 1
                
                try:
                    self.log(f"Processing batch {batch_num}/{total_batches} ({len(batch)} documents)")
                    vector_store.add_documents(batch)
                    self.log(f"Successfully added batch {batch_num}")
                except Exception as e:
                    self.log(f"Error adding batch {batch_num}: {e}")
                    # Continue with next batch instead of failing completely
                    continue
                    
            self.log(f"Completed processing all {len(documents)} documents.")
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

    def search_documents(self, vector_store=None) -> Data:
        """Search documents in the vector store and return formatted results as a single Data object."""
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
            return Data(text="", display_name="No Search Results")

        docs = []
        search_method = "similarity_search"

        try:
            self.log(f"Calling vector_store.{search_method} with args: {search_args}")
            docs = getattr(vector_store, search_method)(**search_args)
        except Exception as e:
            msg = f"Error performing {search_method} in PGVector: {e}"
            raise ValueError(msg) from e

        self.log(f"Retrieved documents: {len(docs)}")

        # Convert documents to a single formatted Data object for parsers
        formatted_data = self._format_search_results_for_parser(docs)
        self.log(f"Formatted documents for parser: {formatted_data.display_name}")
        self.status = formatted_data

        return formatted_data

    def _format_search_results_for_parser(self, docs) -> Data:
        """Format search results for use with Langflow parsers and templates."""
        if not docs:
            return Data(text="", display_name="No Search Results")
        
        # Check if we're in debug mode
        if self.output_template and self.output_template.strip().upper() == "DEBUG":
            return self._debug_mode_output(docs)
        
        # Parse all documents and format them
        parsed_games = []
        
        for i, doc in enumerate(docs):
            # Extract game data from metadata and content
            metadata = doc.metadata or {}
            content = doc.page_content
            
            # Parse the CSV content to extract game data
            game_data = self._parse_csv_content(content, metadata, i + 1)
            parsed_games.append(game_data)
        
        # If using a template, format the results
        self.log(f"Output template value: '{self.output_template}'")
        self.log(f"Template condition check: template='{self.output_template}', strip='{self.output_template.strip() if self.output_template else None}', condition={bool(self.output_template and self.output_template.strip())}")
        
        if self.output_template and self.output_template.strip():
            formatted_results = []
            
            for game_data in parsed_games:
                # Convert lists to strings for template formatting
                mechanics_str = ", ".join(game_data["mechanics"]) if isinstance(game_data["mechanics"], list) else str(game_data["mechanics"])
                domains_str = ", ".join(game_data["domains"]) if isinstance(game_data["domains"], list) else str(game_data["domains"])
                
                try:
                    # Use safe formatting with default values
                    formatted_result = self.output_template.format(
                        id=game_data.get("id", "Unknown"),
                        game_name=game_data.get("game_name", "Unknown"),
                        name=game_data.get("name", "Unknown"),
                        description=game_data.get("description", "No description available"),
                        domains=domains_str,
                        mechanics=mechanics_str,
                        year_published=game_data.get("year_published", "Unknown"),
                        min_players=game_data.get("min_players", "Unknown"),
                        max_players=game_data.get("max_players", "Unknown"),
                        play_time=game_data.get("play_time", "Unknown"),
                        min_age=game_data.get("min_age", "Unknown"),
                        users_rated=game_data.get("users_rated", "Unknown"),
                        rating_average=game_data.get("rating_average", "Unknown"),
                        bgg_rank=game_data.get("bgg_rank", "Unknown"),
                        complexity_average=game_data.get("complexity_average", "Unknown"),
                        owned_users=game_data.get("owned_users", "Unknown"),
                        url=game_data.get("url", "Unknown"),
                        rank=game_data.get("rank", "Unknown"),
                        score=game_data.get("score", "N/A")
                    )
                    formatted_results.append(formatted_result)
                except KeyError as e:
                    self.log(f"Template error for game {game_data.get('game_name', 'Unknown')} - missing key {e}")
                    # Fall back to a simple format
                    simple_result = f"{game_data.get('rank', '?')}. {game_data.get('name', 'Unknown Game')}"
                    formatted_results.append(simple_result)
            
            # Combine all formatted results into a single text
            combined_text = "\n\n".join(formatted_results)
            
            return Data(
                text=combined_text,
                display_name=f"Search Results: {len(formatted_results)} games found",
                metadata={"source": "neon_search", "total_results": len(formatted_results), "template_used": True}
            )
        
        else:
            # Return structured data with individual fields for each game
            self.log("Taking non-template path - returning structured data with individual fields")
            
            # Create a single Data object with all game data as separate fields
            # This allows the parser to access individual fields like rank, name, etc.
            
            # For now, let's return the first game's data as individual fields
            # and put all games' data in the text field
            if parsed_games:
                first_game = parsed_games[0]
                
                # Create a single Data object with individual fields from the first game
                # and all games' data in the text field
                all_games_text = "\n\n".join([
                    f"Game {i+1}: {game['name']} ({game['year_published']}) - {game['rating_average']}/10"
                    for i, game in enumerate(parsed_games)
                ])
                
                return Data(
                    text=all_games_text,
                    rank=first_game.get("rank", 1),
                    id=first_game.get("id", "Unknown"),
                    name=first_game.get("name", "Unknown"),
                    game_name=first_game.get("game_name", "Unknown"),
                    year_published=first_game.get("year_published", "Unknown"),
                    min_players=first_game.get("min_players", "Unknown"),
                    max_players=first_game.get("max_players", "Unknown"),
                    play_time=first_game.get("play_time", "Unknown"),
                    min_age=first_game.get("min_age", "Unknown"),
                    users_rated=first_game.get("users_rated", "Unknown"),
                    rating_average=first_game.get("rating_average", "Unknown"),
                    bgg_rank=first_game.get("bgg_rank", "Unknown"),
                    complexity_average=first_game.get("complexity_average", "Unknown"),
                    owned_users=first_game.get("owned_users", "Unknown"),
                    mechanics=", ".join(first_game.get("mechanics", [])),
                    domains=", ".join(first_game.get("domains", [])),
                    url=first_game.get("url", "Unknown"),
                    description=first_game.get("description", "No description available"),
                    display_name=f"Search Results: {len(parsed_games)} games found",
                    metadata={"source": "neon_search", "total_results": len(parsed_games), "format": "structured"}
                )
            else:
                # No games found
                return Data(
                    text="No games found",
                    rank=0,
                    id="Unknown",
                    name="Unknown",
                    game_name="Unknown",
                    year_published="Unknown",
                    min_players="Unknown",
                    max_players="Unknown",
                    play_time="Unknown",
                    min_age="Unknown",
                    users_rated="Unknown",
                    rating_average="Unknown",
                    bgg_rank="Unknown",
                    complexity_average="Unknown",
                    owned_users="Unknown",
                    mechanics="Unknown",
                    domains="Unknown",
                    url="Unknown",
                    description="No games found",
                    display_name="Search Results: 0 games found",
                    metadata={"source": "neon_search", "total_results": 0, "format": "structured"}
                )

    def _debug_mode_output(self, docs) -> Data:
        """Debug mode to show what data is actually available."""
        debug_info = []
        debug_info.append("=== DEBUG MODE: Available Data Fields ===")
        debug_info.append(f"Found {len(docs)} documents")
        debug_info.append("")
        
        for i, doc in enumerate(docs[:3]):  # Show first 3 documents
            debug_info.append(f"--- Document {i+1} ---")
            debug_info.append(f"Page Content (first 200 chars): {doc.page_content[:200]}...")
            debug_info.append(f"Metadata keys: {list(doc.metadata.keys()) if doc.metadata else 'No metadata'}")
            
            if doc.metadata:
                for key, value in doc.metadata.items():
                    if isinstance(value, (list, dict)):
                        debug_info.append(f"  {key}: {type(value).__name__} with {len(value)} items")
                    else:
                        debug_info.append(f"  {key}: {str(value)[:100]}{'...' if len(str(value)) > 100 else ''}")
            debug_info.append("")
        
        debug_text = "\n".join(debug_info)
        
        return Data(
            text=debug_text,
            display_name="Debug: Data Structure Analysis",
            metadata={"source": "neon_search", "debug_mode": True, "total_docs": len(docs)}
        )

    def _parse_csv_content(self, content, metadata, rank):
        """Parse CSV content to extract game data."""
        import csv
        import io
        
        try:
            # Split the content by commas, but handle quoted fields properly
            reader = csv.reader(io.StringIO(content))
            row = next(reader)
            
            # Expected CSV format: ID,Name,Year,MinPlayers,MaxPlayers,PlayTime,MinAge,UsersRated,Rating,BGGRank,Complexity,OwnedUsers,Mechanics,Domains,URL,Description
            if len(row) >= 16:
                game_data = {
                    "rank": rank,
                    "content": content,
                    "metadata": metadata,
                    "score": None,
                    "id": row[0] if row[0] else "Unknown",
                    "name": row[1] if row[1] else "Unknown",
                    "game_name": row[1] if row[1] else "Unknown",
                    "year_published": row[2] if row[2] else "Unknown",
                    "min_players": row[3] if row[3] else "Unknown",
                    "max_players": row[4] if row[4] else "Unknown",
                    "play_time": row[5] if row[5] else "Unknown",
                    "min_age": row[6] if row[6] else "Unknown",
                    "users_rated": row[7] if row[7] else "Unknown",
                    "rating_average": row[8] if row[8] else "Unknown",
                    "bgg_rank": row[9] if row[9] else "Unknown",
                    "complexity_average": row[10] if row[10] else "Unknown",
                    "owned_users": row[11] if row[11] else "Unknown",
                    "mechanics": row[12].split(", ") if row[12] else [],
                    "domains": row[13].split(", ") if row[13] else [],
                    "url": row[14] if row[14] else "Unknown",
                    "description": row[15] if row[15] else "No description available",
                }
            else:
                # Fallback if CSV format is different
                game_data = {
                    "rank": rank,
                    "content": content,
                    "metadata": metadata,
                    "score": None,
                    "id": "Unknown",
                    "name": "Unknown",
                    "game_name": "Unknown",
                    "year_published": "Unknown",
                    "min_players": "Unknown",
                    "max_players": "Unknown",
                    "play_time": "Unknown",
                    "min_age": "Unknown",
                    "users_rated": "Unknown",
                    "rating_average": "Unknown",
                    "bgg_rank": "Unknown",
                    "complexity_average": "Unknown",
                    "owned_users": "Unknown",
                    "mechanics": [],
                    "domains": [],
                    "url": "Unknown",
                    "description": content,
                }
                
        except Exception as e:
            self.log(f"Error parsing CSV content: {e}")
            # Fallback to basic parsing
            game_data = {
                "rank": rank,
                "content": content,
                "metadata": metadata,
                "score": None,
                "id": "Unknown",
                "name": "Unknown",
                "game_name": "Unknown",
                "year_published": "Unknown",
                "min_players": "Unknown",
                "max_players": "Unknown",
                "play_time": "Unknown",
                "min_age": "Unknown",
                "users_rated": "Unknown",
                "rating_average": "Unknown",
                "bgg_rank": "Unknown",
                "complexity_average": "Unknown",
                "owned_users": "Unknown",
                "mechanics": [],
                "domains": [],
                "url": "Unknown",
                "description": content,
            }
        
        return game_data

    def get_retriever_kwargs(self):
        """Get retriever kwargs for the vector store."""
        search_args = self._build_search_args()

        return {
            "search_type": self._map_search_type(),
            "search_kwargs": search_args,
        }
