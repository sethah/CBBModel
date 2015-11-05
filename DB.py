import psycopg2

class DB(object):

    database = "cbb"
    user = "sethhendrickson"
    password = "abc123"
    host = "localhost"
    port = "5432"
    conn = psycopg2.connect(database=database, user=user, password=password, host=host, port=port)
