import toml
import os
import json
import subprocess
import datetime

CHECKPOINT_DIR = "outputs/"
INITIAL_CHECKPOINT_NAME = "step.safetensors"
BATCH_SIZE = 12
max_tres = 48
PARTITION_NAME="big_suma_rtx3090"
WEIGHT_NAME_FORMAT = "{now}_delta_lokr_bs{BATCH_SIZE}_multinode_l2"

def get_datetime(filename):
    return filename.split("_")[0]

def get_newest_checkpoint():
    return CHECKPOINT_DIR + sorted([f for f in os.listdir(CHECKPOINT_DIR) if f.endswith(".safetensors")], key=lambda x: (get_datetime(x), int(x.split("-")[-1].split(".")[0].lstrip("step").lstrip("0"))))[-1]

def get_sbatch_capability(partition_name, max_tres):
    """
    Returns the capability of the partition_name
    returns nodes, cpus-per-gpu, gres
    """
    sbatch_string = subprocess.run(["python", "auto_qos.py", "--sbatch", f"--max_tres={max_tres}","--partition={}".format(partition_name)], stdout=subprocess.PIPE).stdout.decode("utf-8")
    return sbatch_string.split("\n")

def read_and_replace_lines(filename, partition_name, max_tres):
    with open(filename, "r") as f:
        lines = f.readlines()
    replacements = get_sbatch_capability(partition_name, max_tres)
    replaced_success = [False, False, False]
    for i, line in enumerate(lines):
        # find --nodes= string, --cpus-per-gpu= string, --gres= string and replace them
        if "--nodes=" in line and not replaced_success[0]:
            lines[i] = replacements[0] + "\n"
            node_count = int(replacements[0].split("=")[1])
            replaced_success[0] = True
        elif "--cpus-per-gpu=" in line and not replaced_success[1]:
            lines[i] = replacements[1]+ "\n"
            replaced_success[1] = True
        elif "--gres=" in line and not replaced_success[2]:
            lines[i] = replacements[2]+ "\n"
            gpu_count = int(replacements[2].split(":")[1])
            replaced_success[2] = True
    total_batch_size = BATCH_SIZE * gpu_count
    with open(filename, "w") as f:
        f.writelines(lines)
    print(f"Updated the sbatch capability with {node_count} nodes, {gpu_count} gpus, {total_batch_size} batch size")
    return total_batch_size

def replace_checkpoint_name(filename, batch_size):
    # with toml
    with open(filename, "r") as f:
        config = toml.load(f)
    checkpoint_name = get_newest_checkpoint()
    config["training_arguments"]["network_weights"] = checkpoint_name
    # update extra_arguments.wandb_run_name and extra_arguments.output_name
    datetime_str = datetime.datetime.now().strftime("%m%d%H%M")
    config["extra_arguments"]["wandb_run_name"] = WEIGHT_NAME_FORMAT.format(now=datetime_str, BATCH_SIZE=batch_size)
    config["extra_arguments"]["output_name"] = WEIGHT_NAME_FORMAT.format(now=datetime_str, BATCH_SIZE=batch_size)
    with open(filename, "w") as f:
        toml.dump(config, f)
    print(f"Updated the checkpoint name to {checkpoint_name}")

def check_previous_job_status():
    # check if the previous job is finished
    if os.path.exists("auto_rerun.txt"):
        with open("auto_rerun.txt", "r") as f:
            # get last line
            rerun_job_id = f.readlines()[-1]
        print(f"Rerunning job {rerun_job_id}")
        # check the status of the job
        #squeue --job=<job_id>
        job_status = subprocess.run(["squeue", "--job={}".format(rerun_job_id)], stdout=subprocess.PIPE).stdout.decode("utf-8")
        # if the message was not error, then the job is still running
        if "NODES" in job_status and str(rerun_job_id) in job_status:
            print("The previous job is still running, ending script")
            return False
        else:
            print("The previous job is finished, may require rerun")
            return True
    else:
        print("No previous job to rerun")
        return False

def log_job_id(job_id):
    with open("auto_rerun.txt", "a") as f:
        f.write(job_id)

def check_job_logs_if_preemptied(job_id):
    # check if the job was preemptied
    # check the logs of the job
    # we have to read files with *job_id*
    # if the file contains "PREEMPTIED", then the job was preemptied
    # if the file contains "COMPLETED", then the job was completed
    # either way, we have to rerun the job
    print(f"Checking files with job_id {job_id}")
    files = [f for f in os.listdir() if job_id in f]
    for file in files:
        with open(file, "r") as f:
            if "PREEMPTIED" in f.read():
                print("The job was preemptied")
                return True
            elif "COMPLETED" in f.read():
                print("The job was completed")
                return True
    return False # failed from error

def main(bash_file, config_file, force, max_tres):
    # check if the previous job is finished
    if not check_previous_job_status():
        # check if the job was preemptied
        return
    if os.path.exists("auto_rerun.txt"):
        with open("auto_rerun.txt", "r") as f:
            # get last line
            job_id = f.readlines()[-1]
    if not force and not check_job_logs_if_preemptied(job_id):
        return
    batch_size = read_and_replace_lines(bash_file, PARTITION_NAME, max_tres)
    # update the checkpoint name
    replace_checkpoint_name(config_file, batch_size)
    #sbatch <filename>
    job_id = subprocess.run(["sbatch", bash_file], stdout=subprocess.PIPE).stdout.decode("utf-8")
    # Submitted batch job 660060
    print(f"Submitted batch job {job_id}")
    job_id = job_id.split(" ")[-1]
    log_job_id(job_id)

def log_bash_configs(bash_file, config_file, job_id, active):
    if not os.path.exists("auto_rerun_infos.json"):
        infos = {"bash_file": bash_file, "config_file": config_file, "job_id" : job_id, "active" : active}
        with open("auto_rerun_infos.json", "w") as f:
            json.dump(infos, f)
    else:
        with open("auto_rerun_infos.json", "r") as f:
            infos = json.load(f)
        infos["bash_file"] = bash_file
        infos["config_file"] = config_file
        if job_id:
            infos["job_id"] = job_id
        if active:
            infos["active"] = active
        with open("auto_rerun_infos.json", "w") as f:
            json.dump(infos, f)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--bash_file", type=str, help="The bash file to run the job", default=None)
    parser.add_argument("--config_file", type=str, help="The config file to run the job", default=None)
    parser.add_argument("--active", type=bool, help="Whether to activate the auto rerun", default=True)
    parser.add_argument("--job_id", type=str, help="The job id to rerun", default=None)
    parser.add_argument("--force", type=bool, help="Whether to activate the auto rerun", default=False)
    parser.add_argument("--max_tres", type=int, help="The maximum number of GPUs", default=48)
    args = parser.parse_args()
    bash_file = args.bash_file
    config_file = args.config_file
    active = args.active
    job_id = args.job_id
    max_tres = args.max_tres
    if bash_file and config_file:
        #update the json file
        log_bash_configs(bash_file, config_file, job_id, active)
    else:
        # read the json file
        with open("auto_rerun_infos.json", "r") as f:
            infos = json.load(f)
        bash_file = infos["bash_file"]
        config_file = infos["config_file"]
        active = infos["active"]
        job_id = infos["job_id"]
    if not active:
        exit()

    main(bash_file, config_file, args.force,max_tres)
