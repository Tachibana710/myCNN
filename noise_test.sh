#!/bin/bash

# 0.01から0.25まで0.01刻みで実行
for i in $(seq 0 0.005 0.25); do
    ./build/main $i
done
