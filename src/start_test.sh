#!/bin/bash

# 1st argument specifies the number of clients to create
num_processes=$1
sleep_time=2
array=()

# Start the silent client processes
for ((i=0;i<3;i++))
do
  sleep $sleep_time ; CUDA_VISIBLE_DEVICES=0 python main.py train --distributed --client > /dev/null 2>&1 &
  array+=($!)
  echo $i
done
sleep $sleep_time ; CUDA_VISIBLE_DEVICES=0 python main.py train --distributed --client &
echo $i

sleep $sleep_time ; CUDA_VISIBLE_DEVICES=0 python main.py train --distributed --master -f configuration/euro-gp2020/covid-4clients-MLP.yml &

sleep 300

# Start the silent client processes
for ((i=0;i<3;i++))
do
  sleep $sleep_time ; CUDA_VISIBLE_DEVICES=1 python main.py train --distributed --client > /dev/null 2>&1 &
  array+=($!)
  echo $i
done
sleep $sleep_time ; CUDA_VISIBLE_DEVICES=1 python main.py train --distributed --client &
echo $i
sleep $sleep_time ; CUDA_VISIBLE_DEVICES=1 python main.py train --distributed --master -f configuration/euro-gp2020/covid-4clients-MLP-2.yml

sleep $sleep_time ; CUDA_VISIBLE_DEVICES=0 python main.py train --distributed --master -f configuration/euro-gp2020/covid-4clients-MLP.yml &
sleep 300
sleep $sleep_time ; CUDA_VISIBLE_DEVICES=1 python main.py train --distributed --master -f configuration/euro-gp2020/covid-4clients-MLP-2.yml

sleep $sleep_time ; CUDA_VISIBLE_DEVICES=0 python main.py train --distributed --master -f configuration/euro-gp2020/covid-4clients-MLP.yml &
sleep 300
sleep $sleep_time ; CUDA_VISIBLE_DEVICES=1 python main.py train --distributed --master -f configuration/euro-gp2020/covid-4clients-MLP-2.yml

sleep $sleep_time ; CUDA_VISIBLE_DEVICES=0 python main.py train --distributed --master -f configuration/euro-gp2020/covid-4clients-MLP.yml &
sleep 300
sleep $sleep_time ; CUDA_VISIBLE_DEVICES=1 python main.py train --distributed --master -f configuration/euro-gp2020/covid-4clients-MLP-2.yml

sleep $sleep_time ; CUDA_VISIBLE_DEVICES=0 python main.py train --distributed --master -f configuration/euro-gp2020/covid-4clients-MLP.yml &
sleep 300
sleep $sleep_time ; CUDA_VISIBLE_DEVICES=1 python main.py train --distributed --master -f configuration/euro-gp2020/covid-4clients-MLP-2.yml

sleep $sleep_time ; CUDA_VISIBLE_DEVICES=0 python main.py train --distributed --master -f configuration/euro-gp2020/covid-4clients-MLP.yml &
sleep 300
sleep $sleep_time ; CUDA_VISIBLE_DEVICES=1 python main.py train --distributed --master -f configuration/euro-gp2020/covid-4clients-MLP-2.yml


sleep 60 ; CUDA_VISIBLE_DEVICES=0 python main.py train --distributed --client &
sleep $sleep_time ; CUDA_VISIBLE_DEVICES=1 python main.py train --distributed --master -f configuration/euro-gp2020/covid-9clients-MLP.yml
sleep $sleep_time ; CUDA_VISIBLE_DEVICES=1 python main.py train --distributed --master -f configuration/euro-gp2020/covid-9clients-MLP.yml
sleep $sleep_time ; CUDA_VISIBLE_DEVICES=1 python main.py train --distributed --master -f configuration/euro-gp2020/covid-9clients-MLP.yml
sleep $sleep_time ; CUDA_VISIBLE_DEVICES=1 python main.py train --distributed --master -f configuration/euro-gp2020/covid-9clients-MLP.yml
sleep $sleep_time ; CUDA_VISIBLE_DEVICES=1 python main.py train --distributed --master -f configuration/euro-gp2020/covid-9clients-MLP.yml
sleep $sleep_time ; CUDA_VISIBLE_DEVICES=1 python main.py train --distributed --master -f configuration/euro-gp2020/covid-9clients-MLP.yml
