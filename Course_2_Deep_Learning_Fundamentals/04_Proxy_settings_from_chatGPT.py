import os

print("Hallo, Welt.")
os.environ["http_proxy"] = "http://sia-lb.telekom.de:8080"
os.environ["https_proxy"] = "http://sia-lb.telekom.de:8080"
print("Environment updated:")
print("http_proxy:", os.environ.get("http_proxy"))
print("https_proxy:", os.environ.get("https_proxy"))
