import sqlite3
from sqlite3 import Connection
from langgraph.store.sqlite import SqliteStore

class SyncStoreService:
    def __init__(self, connection: Connection):
        self.connection = connection
        self.store = SqliteStore(conn=self.connection)

        self.connection

    def get_store(self) -> SqliteStore:
        return self.store