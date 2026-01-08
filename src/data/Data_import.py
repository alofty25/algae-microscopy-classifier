import kagglehub

# Download latest version
path = kagglehub.dataset_download("marquis03/high-throughput-algae-cell-detection")

print("Path to dataset files:", path)