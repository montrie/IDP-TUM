#!/bin/bash

# https://stackoverflow.com/questions/17385794/how-to-get-the-process-id-to-kill-a-nohup-process
kill -9 `cat pid.txt`
rm pid.txt
