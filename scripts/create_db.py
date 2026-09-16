"""
Run once to create the PostgreSQL database and all tables.
Usage: python scripts/create_db.py
"""
import asyncio
import sys
sys.path.insert(0, ".")

from app.db.engine import create_all_tables
from app.logging_config import configure_logging, get_logger

configure_logging()
log = get_logger("create_db")


async def main():
    log.info("creating_tables")
    await create_all_tables()
    log.info("tables_created")


if __name__ == "__main__":
    asyncio.run(main())
