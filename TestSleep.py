def main(seconds):
    import time
    print(f"Sleeping for {seconds}...")
    time.sleep(seconds)
    return f"Slept {seconds}"
