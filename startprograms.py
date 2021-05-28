import subprocess
names = ['BFC', 'CY', 'LI', 'LL', 'BF', 'TF']

#process_list = [subprocess.Popen(['python3', 'Main_Shapley_V2.py', name]) for name in names]

for name in names:
    subprocess.Popen(['python3', 'Main_Shapley_V2.py', name]).wait()