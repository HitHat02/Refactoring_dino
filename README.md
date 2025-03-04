
source /home/gpuadmin/anaconda3/etc/profile.d/conda.sh
conda activate py36_torch_HIG

python -m apply_trainer.py

scp -r D:\work_space\data\_rd3_data_all\2022년\서울시\7회차 gpuadmin@180.70.21.13:/home/gpuadmin/ingyu_data/rd3_data/

watch -d -n 0.5 nvidia-smi

git pull origin predict_linux

scp -r gpuadmin@180.70.21.13:/home/gpuadmin/ingyu_data/csv_result E:\csv_from_server\2022_서울시\6회차

mv /home/gpuadmin/ingyu_data/rd3_data/4211-3 /home/gpuadmin/ingyu_data/rd3_data/4회차/4211-3