import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder

def preprocess(df):
    df = df.copy()

    le = LabelEncoder()
    df['Urgency Level'] = le.fit_transform(df['Urgency Level'])
    df['Time of Day'] = le.fit_transform(df['Time of Day'])

    features = [
        'Urgency Level',
        'Nurse-to-Patient Ratio',
        'Specialist Availability',
        'Facility Size (Beds)',
        'Time to Registration (min)',
        'Time to Triage (min)',
        'Time to Medical Professional (min)'
    ]

    X = df[features]
    y = df['Total Wait Time (min)']

    return X, y


def train_models(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    lr = LinearRegression()
    lr.fit(X_train, y_train)

    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)

    return lr, rf


def cluster_analysis(df):
    features = df[['Nurse-to-Patient Ratio', 'Total Wait Time (min)']]

    kmeans = KMeans(n_clusters=3, random_state=42)
    df['Cluster'] = kmeans.fit_predict(features)

    return df, kmeans
