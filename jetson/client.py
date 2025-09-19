# pip install requests
import argparse, sys, time
import requests

def main():
    ap = argparse.ArgumentParser(description="Test /segment endpoint")
    ap.add_argument("--url", default="http://localhost:8000/segment", help="API endpoint")
    ap.add_argument("--image", required=True, help="Path to input image")
    ap.add_argument("--out", default="mask.png", help="Where to save returned mask")
    ap.add_argument("--timeout", type=float, default=60, help="HTTP timeout (s)")
    args = ap.parse_args()

    
    with open(args.image, "rb") as f:
        files = {"image": (args.image, f, "image/*")}
        t0 = time.time()
        r = requests.post(args.url, files=files, timeout=args.timeout)
        dur = time.time() - t0

    if not r.ok:
        print(f"HTTP {r.status_code}: {r.text}", file=sys.stderr)
        sys.exit(1)

    with open(args.out, "wb") as w:
        w.write(r.content)
    print(f"Saved mask -> {args.out}  (elapsed: {dur:.3f}s)")

if __name__ == "__main__":
    main()
