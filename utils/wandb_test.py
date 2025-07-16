import os
import wandb

# Make absolutely sure W&B uses the right Python
os.environ["WANDB_EXECUTABLE"] = "/home/sualpster/miniconda3/envs/project/bin/python"

print("DEBUG sys.executable =", __import__("sys").executable)

# Init under the team where the run/project lives (as shown in your logs)
run = wandb.init(
    project="financial-sentiment-bert",
    entity="cbrkcan90-ludwig-maximilianuniversity-of-munich",
    job_type="download"
)

# Pull the artifact (adjust if needed; see Step 4 below)
artifact = run.use_artifact(
    "cbrkcan90-ludwig-maximilianuniversity-of-munich/financial-sentiment-bert/model-run_financial_bert:v0",
    type="model",
)

artifact_dir = artifact.download()
print(f"Artifact downloaded to: {artifact_dir}")
run.finish()
