#!/usr/bin/bash

while true ; do
    # date +%-M  returns the current minute and strips leading zeros, we had an issue at the 8-th minute of an hour, where 08 % 15 was interpreted as an octal number due to '08' being the prefix denoting octal numbers
    # therefore bash couldnt interpret the modulo calculation correctly: bg_run.sh: 4: arithmetic expression: expecting EOF: " 08 % 15 "
    while [ $(( $(date +%-M) % 15 )) -ne 0 ] ; do
        sleep 60
    done
    python3 main.py
done
