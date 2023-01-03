import sqlite3
import logger


def create_db(db_name: str):
    conn =sqlite3.connect(db_name)
    # logger.Log_ui.add_log(f'Connected to db "{db_name}"')
    conn.close()


def create_table(db_name: str, table_name: str, fields: tuple):
    conn = sqlite3.connect(db_name)
    conn.execute(f'''CREATE TABLE {table_name}
             (ID INTEGER PRIMARY KEY AUTOINCREMENT,
             DATE TEXT NOT NULL,
             WKT TEXT NOT NULL);''')
    conn.commit()
    conn.close()


def insert_data(db_name: str, tb_name: str, date: str, wkt: str):
    conn = sqlite3.connect(db_name)
    sql = f"INSERT INTO {tb_name} (DATE,WKT) VALUES (?, ?);"
    conn.execute(sql, (date.replace(':', '-'), wkt))
    conn.commit()
    conn.close()
    print("save")


def get_wkt(db_name: str, tb_name: str, date: str):
    conn = sqlite3.connect(db_name)
    for row in conn.execute(f"SELECT WKT from {tb_name} WHERE {tb_name}.DATE = '{date}';"):
        return row[0]




