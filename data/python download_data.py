import os
import subprocess

datasets = {
    "playground-series-s4e12": (
        "competition",
        "playground-series-s4e12"
    ),
    "insurance-claims": (
        "dataset",
        "mexwell/insurance-claims"
    )
}

for folder, (kind, identifier) in datasets.items():

    output_dir = os.path.join("data", folder)
    os.makedirs(output_dir, exist_ok=True)

    if kind == "competition":
        command = [
            "kaggle",
            "competitions",
            "download",
            identifier,
            "-p",
            output_dir,
            "--unzip"
        ]
    else:
        command = [
            "kaggle",
            "datasets",
            "download",
            identifier,
            "-p",
            output_dir,
            "--unzip"
        ]

    print(f"\nDownloading {folder}...")
    subprocess.run(command, check=True)

print("\nAll datasets downloaded successfully!")
