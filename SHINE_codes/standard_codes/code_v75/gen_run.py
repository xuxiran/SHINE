seeds = [42,43,1209,1998,2025,2026,10043,110120,123456,125132,213457,321081]
for i in range(12):
    run_file = "run{}.slurm".format(i)
    seed = seeds[i]+56
    valid_num = i
    with open(run_file, "w") as f:
        f.write("#!/bin/bash\n")
        f.write("#SBATCH -o ./log/%j.out\n")
        f.write("#SBATCH -e ./log/%j.err\n")
        f.write("#SBATCH --ntasks-per-node=8\n")
        f.write("#SBATCH --partition=GPUA800\n")
        f.write("#SBATCH --exclude=gpua800n22,gpua800n25,gpua800n13\n")
        f.write("#SBATCH -J v56_{}\n".format(i))
        f.write("#SBATCH --gres=gpu:1\n\n")
        # f.write("python main.py --seed {} --model_name {}\n".format(seed,model))
        f.write("python main.py --seed {} --valid_num {}\n".format(seed,valid_num))



