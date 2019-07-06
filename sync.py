# simple script to sync run folder on linux to Dropbox

import time
import os

while True:

    print("-"*60)
    print("Quick Sync:")

    # run quick sync (ignoring model and video files)
    os.system("./dropbox_uploader.sh -x models -x videos upload runs /")

    print("-" * 60)
    print("Slow Sync:")

    # run slow sync (everything, but without overwriting)
    os.system("./dropbox_uploader.sh -s upload runs /")

    print("Waiting 5 minutes...")

    # wait 5 minutes...
    time.sleep(60*5)