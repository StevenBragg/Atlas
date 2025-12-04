"""Check Atlas database contents."""
import sqlite3
import os

DATA_DIR = r'C:\Users\sabragg\Desktop\self_learn\Atlas\atlas_data'

print("=" * 60)
print("ATLAS DATABASE STATUS CHECK")
print("=" * 60)

# Check Graph DB
graph_db = os.path.join(DATA_DIR, 'atlas_graph.db')
if os.path.exists(graph_db):
    print(f"\n[GRAPH DB] {graph_db}")
    print(f"  File size: {os.path.getsize(graph_db):,} bytes")
    conn = sqlite3.connect(graph_db)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = cursor.fetchall()
    for t in tables:
        cursor.execute(f'SELECT COUNT(*) FROM {t[0]}')
        count = cursor.fetchone()[0]
        print(f"  Table '{t[0]}': {count} rows")
        if count > 0 and count <= 5:
            cursor.execute(f'SELECT * FROM {t[0]} LIMIT 3')
            rows = cursor.fetchall()
            for row in rows:
                print(f"    -> {row[:3]}...")  # First 3 columns
    conn.close()
else:
    print(f"\n[GRAPH DB] NOT FOUND: {graph_db}")

# Check Network DB
network_db = os.path.join(DATA_DIR, 'atlas_network.db')
if os.path.exists(network_db):
    print(f"\n[NETWORK DB] {network_db}")
    print(f"  File size: {os.path.getsize(network_db):,} bytes")
    conn = sqlite3.connect(network_db)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = cursor.fetchall()
    for t in tables:
        cursor.execute(f'SELECT COUNT(*) FROM {t[0]}')
        count = cursor.fetchone()[0]
        print(f"  Table '{t[0]}': {count} rows")
    conn.close()
else:
    print(f"\n[NETWORK DB] NOT FOUND: {network_db}")

# Check ChromaDB
chroma_dir = os.path.join(DATA_DIR, 'chroma')
if os.path.exists(chroma_dir):
    print(f"\n[CHROMADB] {chroma_dir}")
    total_size = 0
    file_count = 0
    for root, dirs, files in os.walk(chroma_dir):
        for f in files:
            fp = os.path.join(root, f)
            total_size += os.path.getsize(fp)
            file_count += 1
    print(f"  Total files: {file_count}")
    print(f"  Total size: {total_size:,} bytes")

    # Try to query ChromaDB
    try:
        import chromadb
        client = chromadb.PersistentClient(path=chroma_dir)
        collections = client.list_collections()
        print(f"  Collections: {len(collections)}")
        for coll in collections:
            count = coll.count()
            print(f"    - '{coll.name}': {count} items")
    except Exception as e:
        print(f"  ChromaDB query error: {e}")
else:
    print(f"\n[CHROMADB] NOT FOUND: {chroma_dir}")

print("\n" + "=" * 60)
