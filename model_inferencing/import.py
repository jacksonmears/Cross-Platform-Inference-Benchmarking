import os

file_path = "raw_scans"
files = [file for file in os.listdir(file_path)]

for file in files:
    full_path = os.path.join(file_path, file)

    with open(full_path, 'r') as f:
        lines = f.readlines()

    new_lines = [line.replace(',',' ') for line in lines]
    new_path = "cleaned_scans/"+file[:-4]+"_cleaned.xyz"

    print(full_path, new_path, len(new_lines))

    with open(new_path, 'w') as f:
        f.writelines(new_lines)

    print(f"Processed {new_path}")
