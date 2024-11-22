# deploy.py
import glob

def deploy():
    model_files = glob.glob("model_*.pt")
    if model_files:
        latest_model = max(model_files, key=lambda x: x)
        print(f"Deploying model: {latest_model}")
    else:
        print("No model found to deploy.")

if __name__ == "__main__":
    deploy()