#!/mnt/fast/nobackup/users/ae01116/multi-modal-dissertation-uos/zegenv/bin/python3

import requests
import sys
import argparse
import time

def parse_args():
    parser = argparse.ArgumentParser(description="Link my telegram to an output file")
    parser.add_argument("--job", help="train config file path")
    parser.add_argument("--type", help="error for training and out for testing", default="error")
    parser.add_argument("--time", help="time between two updates", default=2*60, type=int)
    args = parser.parse_args()
    return args

def send_message(message):
    TOKEN = "7298020171:AAHw7QaqULqbNMXAstprMB_oS97SJDJ_-tg"
    chat_id = "6284547224"
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage?chat_id={chat_id}&text={message}"
    print(requests.get(url).json())

def open_file(file):
    try:
        with open(file, "r") as f:
            file_content = f.read()
    except FileNotFoundError:
        send_message(f"File {file} not found.")
        file_content = None
    return file_content



if __name__ == "__main__":
    args = parse_args()
    job = args.job
    _type = args.type
    _time = args.time
    print(f"Watching job {job}")
    file = f"/mnt/fast/nobackup/users/ae01116/multi-modal-dissertation-uos/logs/c{job}.p0.{_type}"
    # every 30 seconds
    send_message("----- Starting to watch the file -----\n")
    send_message(f"Watching file {file}")
    old_content = open_file(file)
    if old_content == None:
        sys.exit(0)
    while True:
        time.sleep(_time)
        # for every new line of the file, send a message
        with open(file, "r") as f:
            content = f.read()
            message = content[len(old_content):]
            if len(message) > 0:
                a=0
                for line in message.split("\n"):
                    send_message(line)
                    a+=1
                    if a>10:
                        break
                old_content = content
            else:
                send_message("No new lines, stopping the watch.")
                break
        