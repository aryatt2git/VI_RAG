import json
import weaviate

client = weaviate.connect_to_custom(
    http_host="localhost",
    http_port=5050,
    http_secure=False,
    grpc_host="localhost",
    grpc_port=50051,
    grpc_secure=False,
)

try:
    client.is_ready()

    # This prints out the available AIs to API to in the docker-compose.yml
    # Change the 'ENABLE_MODULES' and 'DEFAULT_VECTORIZER_MODULE' modules in docker-compose.yml as well as the API key
    # in weaviate.env.
    """
    metainfo = client.get_meta()
    print(json.dumps(metainfo, indent=2))
    """

    response = client._connection.get(path="/nodes?output=verbose")
    raw_data = response.json()

    print("--- RAW NODE STATUS ---")
    for node in raw_data.get("nodes", []):
        name = node.get("name")
        status = node.get("status")

        # 2. Look for the 'stats' key in the raw JSON
        stats = node.get("stats", {})
        disk = stats.get("diskUsage", {})  # Note: API uses camelCase

        if disk:
            # Weaviate API sends bytes, let's convert to GB
            total = disk.get("total", 0) / (1024 ** 3)
            used = disk.get("used", 0) / (1024 ** 3)
            percent = (used / total * 100) if total > 0 else 0

            print(f"Node: {name} | Status: {status}")
            print(f"  > Disk Used:  {used:.2f} GB")
            print(f"  > Disk Total: {total:.2f} GB")
            print(f"  > Usage:      {percent:.1f}%")
        else:
            print(f"Node: {name} | No disk stats found in JSON.")
            # Print the stats object to see what's actually there
            print(f"  > Available stats keys: {list(stats.keys())}")

finally:
    client.close()