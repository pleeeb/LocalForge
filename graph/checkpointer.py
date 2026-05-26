from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3
from sqlite3 import Connection

class SqliteDatabase:
    def __init__(self, db_path: str):
        self.db_path = db_path

        self.connection = sqlite3.connect(self.db_path, check_same_thread=False)

    def get_connection(self) -> Connection:
        return self.connection

class SyncCheckpointer:
    def __init__(self, connection: Connection):
        self.connection = connection
        self.saver = SqliteSaver(conn=self.connection)

        self.saver.setup()

    def get_checkpointer(self) -> SqliteSaver:
        return self.saver
    
