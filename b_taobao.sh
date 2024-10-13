#!/bin/bash


# lr=('0.001' '0.005' '0.01')
# lr=('0.001' '0.0001' '0.0003' '0.0005')
lr=('0.001')
# lr=('0.01')
# reg_weight=('0.01' '0.001' '0.0001')
emb_size=(64)
# lr=('0.0005')
reg_weight=('0.0001')
# log_regs=('0.5' '0.7' '1.0')
# log_regs=('0.9' '0.7' '0.5')
# log_regs=('1.0' '1.2' '1.5')

layers=('2')

# num_experts=('2')
# k=('1' '3' '5' '7' '9' '11' '13' '15' '16')
# k=('1' '3' '5')
# num_experts=('2' '4' '6' '8' '10' '12' '14' '16')
# k=('16')
 num_experts=('10')
#num_experts=('18' '20' '22' '24' '26' '28' '30' '32')

# ssl_tau=('0.1' '0.3' '0.5' '0.7' '0.9')
ssl_tau=('0.7')
ssl_reg=0.1

dataset=('taobao')
device='cuda:3'
batch_size=1024
decay=('0')

model_name='model'
log_name='model'


gpu_no=1
neg_count=4


for name in ${dataset[@]}
do
    for l in ${lr[@]}
    do
        for reg in ${reg_weight[@]}
        do
            for emb in ${emb_size[@]}
            do
            for dec in ${decay[@]}
            do
            for export in ${num_experts[@]}
            do
            for gcn in ${layers[@]}
            do
            for t in ${ssl_tau[@]}
            do

                echo 'start train: '$name
                `
                    python main.py \
                        --model_name $model_name \
                        --log_name $log_name \
                        --lr ${l} \
                        --ssl_reg $ssl_reg \
                        --reg_weight ${reg} \
                        --num_experts ${export} \
                        --data_name $name \
                        --embedding_size $emb \
                        --device $device \
                        --decay $dec \
                        --batch_size $batch_size \
                        --gpu_no $gpu_no \
                        --layers ${gcn} \
                        --ssl_tau ${t}
                `
                echo 'train end: '$name

            done
            done
            done
            done
            done
        done
    done
done