"""Parallel (multi-connection) downloader for the CIFAR-10 python tarball.

The cs.toronto.edu server throttles per connection (~10 KB/s), so we fetch the
file in N byte-ranges concurrently to bypass that limit. Verifies md5 at the end.

Usage (on the cluster):
    python cluster/parallel_download.py
"""

import hashlib
import os
import threading
import time
import urllib.request

URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
OUT = "/work/scratch/rbraganca/dynfednas/data/cifar10/cifar-10-python.tar.gz"
EXPECT_MD5 = "c58f30108f718f92721af3b95e74349a"
N = 16  # parallel connections


def main():
    head = urllib.request.Request(URL, method="HEAD")
    size = int(urllib.request.urlopen(head).headers["Content-Length"])
    print("total size:", size, flush=True)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "wb") as f:
        f.truncate(size)

    part = size // N
    done = [0]
    lock = threading.Lock()

    def fetch(i):
        start = i * part
        end = size - 1 if i == N - 1 else start + part - 1
        req = urllib.request.Request(URL, headers={"Range": "bytes=%d-%d" % (start, end)})
        data = urllib.request.urlopen(req).read()
        with open(OUT, "r+b") as f:
            f.seek(start)
            f.write(data)
        with lock:
            done[0] += 1
            print("part %d/%d done" % (done[0], N), flush=True)

    t0 = time.time()
    threads = [threading.Thread(target=fetch, args=(i,)) for i in range(N)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    md5 = hashlib.md5(open(OUT, "rb").read()).hexdigest()
    ok = "OK" if md5 == EXPECT_MD5 else "MISMATCH"
    print(
        "done in %.0fs  size=%d  md5=%s  %s"
        % (time.time() - t0, os.path.getsize(OUT), md5, ok),
        flush=True,
    )


if __name__ == "__main__":
    main()
