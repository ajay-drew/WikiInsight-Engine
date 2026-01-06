"""
Database connection management for WikiInsight Engine.

Provides async connection pool using asyncpg and SQLAlchemy.
Supports both PostgreSQL with pgvector and fallback modes.
"""

import logging
import os
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Optional

import yaml
from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.pool import NullPool

from .models import Base

logger = logging.getLogger(__name__)

# Global database manager instance
_db_manager: Optional["DatabaseManager"] = None


class DatabaseManager:
    """
    Manages database connections and sessions.
    
    Provides async connection pooling and session management
    for PostgreSQL with pgvector support.
    """
    
    def __init__(
        self,
        database_url: Optional[str] = None,
        pool_size: int = 10,
        echo: bool = False,
    ):
        """
        Initialize database manager.
        
        Args:
            database_url: PostgreSQL connection URL (async format)
            pool_size: Connection pool size
            echo: Enable SQL logging
        """
        self.database_url = database_url or self._get_database_url()
        self.pool_size = pool_size
        self.echo = echo
        self._engine = None
        self._session_factory = None
        self._initialized = False
    
    def _get_database_url(self) -> str:
        """Get database URL from environment or config."""
        # Try environment variable first
        url = os.environ.get("DATABASE_URL")
        if url:
            # Convert postgres:// to postgresql+asyncpg://
            if url.startswith("postgres://"):
                url = url.replace("postgres://", "postgresql+asyncpg://", 1)
            elif url.startswith("postgresql://"):
                url = url.replace("postgresql://", "postgresql+asyncpg://", 1)
            return url
        
        # Try config file
        config_path = "config.yaml"
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                config = yaml.safe_load(f) or {}
            
            db_config = config.get("database", {})
            url = db_config.get("url")
            if url:
                if url.startswith("postgres://"):
                    url = url.replace("postgres://", "postgresql+asyncpg://", 1)
                elif url.startswith("postgresql://") and "+asyncpg" not in url:
                    url = url.replace("postgresql://", "postgresql+asyncpg://", 1)
                return url
        
        # Default for development
        return "postgresql+asyncpg://postgres:postgres@localhost:5432/wikiinsight"
    
    async def initialize(self) -> None:
        """Initialize database engine and create tables."""
        if self._initialized:
            return
        
        logger.info("Initializing database connection...")
        logger.info("Database URL: %s", self._mask_password(self.database_url))
        
        # Create async engine
        self._engine = create_async_engine(
            self.database_url,
            echo=self.echo,
            pool_size=self.pool_size,
            max_overflow=self.pool_size * 2,
            pool_pre_ping=True,
        )
        
        # Create session factory
        self._session_factory = async_sessionmaker(
            bind=self._engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )
        
        # Create tables
        async with self._engine.begin() as conn:
            # Enable pgvector extension
            try:
                await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
                logger.info("pgvector extension enabled")
            except Exception as e:
                logger.warning("Could not enable pgvector extension: %s", e)
            
            # Enable trigram extension for text search
            try:
                await conn.execute(text("CREATE EXTENSION IF NOT EXISTS pg_trgm"))
                logger.info("pg_trgm extension enabled")
            except Exception as e:
                logger.warning("Could not enable pg_trgm extension: %s", e)
            
            # Create all tables
            await conn.run_sync(Base.metadata.create_all)
            logger.info("Database tables created")
        
        self._initialized = True
        logger.info("Database initialization complete")
    
    async def close(self) -> None:
        """Close database connections."""
        if self._engine:
            await self._engine.dispose()
            self._engine = None
            self._session_factory = None
            self._initialized = False
            logger.info("Database connections closed")
    
    @asynccontextmanager
    async def session(self) -> AsyncGenerator[AsyncSession, None]:
        """
        Get an async database session.
        
        Usage:
            async with db_manager.session() as session:
                result = await session.execute(query)
        """
        if not self._initialized:
            await self.initialize()
        
        async with self._session_factory() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise
    
    async def health_check(self) -> bool:
        """Check if database connection is healthy."""
        try:
            async with self.session() as session:
                result = await session.execute(text("SELECT 1"))
                return result.scalar() == 1
        except Exception as e:
            logger.error("Database health check failed: %s", e)
            return False
    
    async def get_stats(self) -> dict:
        """Get database statistics."""
        try:
            async with self.session() as session:
                # Count articles
                result = await session.execute(text("SELECT COUNT(*) FROM articles"))
                article_count = result.scalar() or 0
                
                # Count clusters
                result = await session.execute(text("SELECT COUNT(*) FROM clusters"))
                cluster_count = result.scalar() or 0
                
                # Count edges
                result = await session.execute(text("SELECT COUNT(*) FROM graph_edges"))
                edge_count = result.scalar() or 0
                
                return {
                    "articles": article_count,
                    "clusters": cluster_count,
                    "edges": edge_count,
                    "connected": True,
                }
        except Exception as e:
            logger.error("Failed to get database stats: %s", e)
            return {
                "articles": 0,
                "clusters": 0,
                "edges": 0,
                "connected": False,
                "error": str(e),
            }
    
    @staticmethod
    def _mask_password(url: str) -> str:
        """Mask password in database URL for logging."""
        if "@" in url and "://" in url:
            # Extract parts
            protocol_end = url.index("://") + 3
            at_pos = url.index("@")
            
            # Find password start (after colon in credentials)
            creds = url[protocol_end:at_pos]
            if ":" in creds:
                colon_pos = creds.index(":")
                user = creds[:colon_pos]
                return f"{url[:protocol_end]}{user}:****{url[at_pos:]}"
        
        return url


def get_db_manager() -> DatabaseManager:
    """Get or create the global database manager instance."""
    global _db_manager
    if _db_manager is None:
        _db_manager = DatabaseManager()
    return _db_manager


async def init_database() -> DatabaseManager:
    """Initialize and return the database manager."""
    manager = get_db_manager()
    await manager.initialize()
    return manager

