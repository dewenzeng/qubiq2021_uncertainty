#!/bin/bash 
#JSUB -J qubiq2021
#JSUB -q tensorflow_sub
#JSUB -gpgpu "num=1"
#JSUB -R "span[ptile=1]" 
#JSUB -n 1
#JSUB -o logs/output.%J 
#JSUB -e logs/err.%J 
##########################Cluster environment variable###################### 
if [ -z "$LSB_HOSTS" -a -n "$JH_HOSTS" ]
then
        for var in ${JH_HOSTS}
        do
                if ((++i%2==1))
                then
                        hostnode="${var}"
                else
                        ncpu="$(($ncpu + $var))"
                        hostlist="$hostlist $(for node in $(seq 1 $var);do printf "%s " $hostnode;done)"
                fi
        done
        export LSB_MCPU_HOSTS="$JH_HOSTS"
        export LSB_HOSTS="$(echo $hostlist|tr ' ' '\n')"
fi

nodelist=.hostfile.$$
for i in `echo $LSB_HOSTS`
do
    echo "${i}" >> $nodelist
done

ncpu=`echo $LSB_HOSTS |wc -w`
##########################Software environment variable#####################
module load python/3.6.10
module load cuda/cuda10.1 
module load pytorch/pytorch1.5.1 

train_base_dir=/data/users/dewenzeng/data/qubiq2021/training_data_v3/
vali_base_dir=/data/users/dewenzeng/data/qubiq2021/validation_data_qubiq2021/
test_base_dir=/data/users/dewenzeng/data/qubiq2021/validation_data_qubiq2021/
python3 main.py --device cuda:0 --batch_size 5 --epochs 400 --train_base_dir ${train_base_dir} --vali_base_dir ${vali_base_dir} --test_base_dir ${test_base_dir} --lr 1e-4 --min_lr 1e-6 --dataset pancreatic-lesion --task task01 \
--initial_filter_size 48 --classes 2 --patch_size 512

# python3 predict.py --device cuda:0 --test_base_dir ${test_base_dir} --dataset kidney --task task01 \
# --initial_filter_size 32 --classes 2 --patch_size 512 --pretrained_model_path /data/users/dewenzeng/code/qubiq2021_uncertainty/results/kidney_task01_2021-08-06_04-51-12/model/latest.pth