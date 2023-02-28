from re import search
from psycopg2 import OperationalError
import psycopg2
import logging
import traceback
import time

class RDSCursor:
    def __init__(self, host, dbname, user, password, table):
        self.host = host
        self.dbname = dbname
        self.user = user
        self.password = password
        self.table = table
    
        connection_config = f'host={self.host} '
        connection_config += f'dbname={self.dbname} '
        connection_config += f'user={self.user} '
        connection_config += f'password={self.password} '
        # connection_config += f'connect_timeout=5'
        self.connection_config = connection_config

        self.connect()
        
    def connect(self):
        self.conn = psycopg2.connect(self.connection_config)
        self.cursor = self.conn.cursor()
        self.connection_time = time.time()

    def clear_all(self):
        self.execute_sql(f'delete from {self.table}')
    
    def copy_from_file(self, filename, table_name):
        with open(filename, 'r') as f:
            self.cursor.copy_from(f, table_name, sep=',')
        self.conn.commit()

    def execute_sql(self, query):
        if time.time() - self.connection_time > 20 * 60:
            self.conn.close()

        if self.conn.closed:
            self.connect()

        try:
            self.cursor.execute(query)
            self.conn.commit()
        except OperationalError:
            logging.error(f'OperationalError caught: {traceback.format_exc()}')
            self.connect()
            self.cursor.execute(query)
            self.conn.commit()
        except:
            logging.error(f'rds_bridge error caught: {traceback.format_exc()}')
            if not self.conn.closed:
                self.conn.rollback()
        return

    def get_all_records(self):
        self.execute_sql(f'SELECT * FROM {self.table}')
        return self.cursor.fetchall()

    def insert_record(self, columns, values):
        self.insert_record_wo_quotes(columns, ['\''+val+'\'' for val in values])

    def insert_record_wo_quotes(self, columns, values):
        column_list = ', '.join(columns)
        
        value_list = ', '.join(val for val in values)
        self.execute_sql(f'INSERT INTO {self.table} ({column_list}) VALUES ({value_list}) ON CONFLICT DO NOTHING')

    def get_column(self, target_col, search_col, value):
        # get target_col of a row with search_col == value
        # value has to be a VARCHAR
        self.execute_sql(f'SELECT {target_col} FROM {self.table} WHERE {search_col}=\'{value}\'')
        return self.cursor.fetchall()

    def get(self, search_col, value):
        # get target_col of a row with search_col == value
        # value has to be a VARCHAR
        self.execute_sql(f'SELECT * FROM {self.table} WHERE {search_col}=\'{value}\'')
        return self.cursor.fetchall()

    def get_column_multi_and(self, search_dict):
        # value is not VARCHAR, it is itself
        query = f'SELECT * FROM {self.table} WHERE '
        for key in search_dict:
            query += f'{key}={search_dict[key]}'
            query += ' AND '
        query = query[:-4]
        self.execute_sql(query)
        return self.cursor.fetchall()
        

    def update_record_by_col(self, target_col, target_val, update_col, update_val):
        # target_val is INT
        # update_val is VARCHAR
        self.execute_sql(f'UPDATE {self.table} SET {update_col} = \'{update_val}\' WHERE {target_col} = {target_val}')
