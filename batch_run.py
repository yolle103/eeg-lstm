import os
import subprocess
import datetime

run_list = [6, 8, 12, 13, 14, 15]
run_script = 'hyper_search_model.py'
start_epoch = 0
epoch = 20
d = datetime.date.today()
date_prefix = '{}_{}'.format(d.month, d.day)

for item in run_list:
    folder = './LOPO_image_slide_3/chb{0:02}'.format(item)
    save_dir = './training_{0}/model_chb{1:02}'.format(date_prefix, item)
    print(folder, save_dir)
    log_file = './{}.log'.format(item)
    with open(log_file, 'w') as f:
        subprocess.call('nohup python {} -f {} -s {} -se {} -e {} & '
            .format(run_script, folder, save_dir, start_epoch, epoch, item), shell=True, stdout=f, stderr=f)
