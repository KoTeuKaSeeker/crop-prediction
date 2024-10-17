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

def generate_directory_for_prefix(dir: str, prefix: str):
    os.makedirs(dir, exist_ok=True)
    last_dir_id = get_last_prefix_dir_id(dir, prefix) + 1
    new_run_dir = os.path.join(dir, prefix + str(last_dir_id))
    os.makedirs(new_run_dir, exist_ok=True)

if __name__ == "__main__":
    generate_directory_for_prefix("runs/crop_transformer/", "phase")
    # run_dir = "runs/crop_transformer/"
    # os.makedirs(run_dir, exist_ok=True)
    # run_dir_folder_names = os.listdir(run_dir)
    # last_run_id = -1
    # for folder_name in run_dir_folder_names:
    #     if bool(re.match(r"^run\d+$", folder_name)):
    #         run_id = int(re.findall(r'\d+', folder_name)[0])
    #         if run_id > last_run_id:
    #             last_run_id = run_id
    # last_run_id += 1
    # new_run_dir = os.path.join(run_dir, "run" + str(last_run_id))
    # os.makedirs(new_run_dir, exist_ok=True)