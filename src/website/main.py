import requests

BASE = "https://jsonplaceholder.typicode.com"

new_post = {
    "title": "hello",
    "body": "world",
    "userId": 1,
}
resp = requests.post(f"{BASE}/posts", json=new_post)
print(resp.status_code)   # 201 Created
print(resp.json())


resp = requests.get(f"{BASE}/posts/101")
resp.raise_for_status()               # explode early if something’s wrong
post = resp.json()                    # dict
print(post["title"])
