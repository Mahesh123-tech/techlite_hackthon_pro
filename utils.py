import pandas as pd
import sqlite3

def load_data(file):
    df = pd.read_csv(file)

    # Convert datetime
    df['Visit Date'] = pd.to_datetime(df['Visit Date'])

    # Feature engineering
    df['Hour'] = df['Visit Date'].dt.hour
    df['Day'] = df['Day of Week']

    return df


def create_sql_db(df):
    conn = sqlite3.connect("er.db")
    df.to_sql("er_table", conn, if_exists="replace", index=False)
    return conn


def run_queries(conn):
    queries = {}

    # 1. Average wait time by hour
    queries['hourly_wait'] = pd.read_sql("""
        SELECT Hour, AVG("Total Wait Time (min)") as avg_wait
        FROM er_table
        GROUP BY Hour
        ORDER BY Hour
    """, conn)

    # 2. Day wise bottlenecks
    queries['day_wait'] = pd.read_sql("""
        SELECT "Day of Week", AVG("Total Wait Time (min)") as avg_wait
        FROM er_table
        GROUP BY "Day of Week"
    """, conn)

    # 3. Staffing impact
    queries['staffing'] = pd.read_sql("""
        SELECT "Nurse-to-Patient Ratio",
               AVG("Total Wait Time (min)") as avg_wait
        FROM er_table
        GROUP BY "Nurse-to-Patient Ratio"
    """, conn)

    return queries
