import datetime

def log_event(event):
    with open("logs.txt", "a") as f:
        time = datetime.datetime.now().strftime("%H:%M:%S")
        f.write(f"{time} - {event}\n")