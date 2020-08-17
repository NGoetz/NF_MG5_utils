#!/usr/bin/env bash
FILE=$1
cd SubProcesses;

for f in *; 
    do   if [ -d "$f" ]; 
        then cd $f; 
        make matrix2py.so 2>&1 > f2py_log.txt 2>&1;
        cd ..;
        cd ..;
        cd ..;
        cd ..;
        cp PLUGIN/check_sa_nis.py  $FILE/SubProcesses/$f/; 
        cd $FILE/SubProcesses/$f;
        echo $f 2>&1 > record.txt 2>&1;
        ./check_sa_nis.py 2>&1 >> record.txt 2>&1;
        cd ..;
    fi
done
