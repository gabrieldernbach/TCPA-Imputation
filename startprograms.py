import subprocess
names = ['BFC', 'CY', 'LI', 'LL', 'BF', 'TF']:

process_list = [subprocess.Popen(['screen python3' name]) for name in names]

for proc in process_list:
    proc.wait()