import yaml
import os
import re


def get_last_prefix_dir_id(dir: str, prefix: str):
    folder_names = os.listdir(dir)
    last_dir_id = -1
    for folder_name in folder_names:
        pattern = r"^{}\d+$".format(prefix)
        if bool(re.match(pattern, folder_name)):
            run_id = int(re.findall(r'\d+', folder_name)[0])
            if run_id > last_dir_id:
                last_dir_id = run_id
    return last_dir_id


if __name__ == "__main__":
    with open("configs/resume_training.yaml", 'r') as file:
        resume_config = yaml.safe_load(file)
    
    source_run_id = int(resume_config["run_id"])
    source_phase_id = int(resume_config["phase_id"])
    if resume_config["run_id"] == 'last':
        run_id = get_last_prefix_dir_id("runs/crop_transformer", "run")




    