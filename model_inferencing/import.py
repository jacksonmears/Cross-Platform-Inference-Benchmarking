import os

script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, "raw_scans")
output_dir = os.path.join(script_dir, "cleaned_scans")

# Create the output folder if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

files = [file for file in os.listdir(file_path)]

for file in files:
    full_path = os.path.join(file_path, file)

    with open(full_path, 'r') as f:
        lines = f.readlines()

    new_lines = [line.replace(',', ' ') for line in lines]
    new_path = os.path.join(output_dir, file[:-4] + "_cleaned.xyz")

    print(full_path, new_path, len(new_lines))

    with open(new_path, 'w') as f:
        f.writelines(new_lines)

    print(f"Processed {new_path}")
