import argparse
import subprocess
import os


parser = argparse.ArgumentParser()
parser.add_argument("--input_images", type=str)
parser.add_argument("--output_dir", type=str)

args = parser.parse_args() 

print(args.input_images)
print(args.output_dir)
output_json = os.path.join(args.output_dir, args.input_images.split("/")[-2] + ".json")

fidelity_args = f"fidelity --gpu 0 --fid --input1 {args.input_images} --input2 cifar10-train --json --silent".split()
with open(output_json, 'w') as f:        
    subprocess.call(fidelity_args, stdout=f)
