model_name="path to expert models"

python panda.py --task_name sentiment --mode gather_pref --data_size 1000 \
            --model_name $model_name --output_path ./logs &