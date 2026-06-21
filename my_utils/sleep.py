import time

def sleep_a_day():
    """Pause execution for one day (86,400 seconds)."""
    time.sleep(86400*2)

# Example usage:
if __name__ == "__main__":
    print("Going to sleep for one day...")
    sleep_a_day()
    print("Woke up after one day!")