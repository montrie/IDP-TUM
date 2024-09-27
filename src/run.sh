#!/usr/bin/bash

# create a background process which doesn't end when closing the terminal that runs main.py and redirects the stdout and stderr outputs to /dev/null, effectively discarding it
# nohup: create process that doesn't end when closing the terminal
# >/dev/null: redirect stdout to /dev/null, effectively discarding the output
# 2>&1: redirect stderr to stdout (i.e. /dev/null), & in &1 means file descriptor, not file name
# &: start process in the background, immediatelly freeing the current terminal
# see https://stackoverflow.com/questions/10408816/how-do-i-use-the-nohup-command-without-getting-nohup-out 
#     https://askubuntu.com/questions/12098/what-does-outputting-to-dev-null-accomplish-in-bash-scripts
#                     >/dev/null 2>&1
nohup sh bg_run.sh> errlog.log 2>&1 </dev/null & 

# $! contains the process ID of the most recently executed background pipeline: https://unix.stackexchange.com/questions/85021/in-bash-scripting-whats-the-meaning-of
# pid.txt can later be used to kill the started background process
# kill -9 `cat ~/IDP-TUM/src/pid.txt`
#  ps aux | grep `cat ~/IDP-TUM/src/pid.txt` to check whether the process is still running -> there should be 2 lines of output if the process is alive
echo $! > pid.txt
