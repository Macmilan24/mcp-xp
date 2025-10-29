import redis

try:
    # Connect to your local Redis instance
    r = redis.Redis(host='localhost', port=6379, db=0)
    
    # This command deletes all keys in the current database
    r.flushdb()
    
    print("✅ Successfully cleared Redis cache (database 0).")

except redis.exceptions.ConnectionError as e:
    print(f"❌ Could not connect to Redis. Is the server running at localhost:6379? Error: {e}")