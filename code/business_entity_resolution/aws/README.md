# Running on AWS (SageMaker Notebook Instance)

Follows the official prep guide (SageMaker **Notebook Instance**, not Studio; region **us-east-1**).

## 0. Before creating anything
1. Set a **billing alert** (Billing → Budgets → $50 and $150 thresholds) on every account.
2. Open **Service Quotas → Amazon SageMaker** and check the quota for
   `ml.g5.xlarge for notebook instance usage` and `ml.g4dn.xlarge for notebook instance usage`.
   New accounts often have 0; request 1 each right away (approval can take hours to a day).
   CPU instances (`ml.m5.4xlarge`, `ml.r5.4xlarge`) normally have quota already.

## 1. Data to S3 (once, from any machine with the AWS CLI)
```
aws s3 mb s3://<bucket>
aws s3 sync student_resource/dataset s3://<bucket>/dataset
aws s3 cp   student_resource/utils/validate_submission.py s3://<bucket>/utils/validate_submission.py
aws s3 sync code/business_entity_resolution s3://<bucket>/code/business_entity_resolution --exclude "__pycache__/*"
```
(~3 GB of TSVs; S3 storage is within the free tier.)

## 2. Create the notebook instance
SageMaker AI console → Notebook → Notebook instances → Create:
- Name: `ber-cpu` (or `ber-gpu`)
- Instance type: `ml.m5.4xlarge` for the pipeline (16 vCPU / 64 GB), `ml.g5.xlarge` or `ml.g4dn.xlarge` for embedding experiments
- **Volume size: 100 GB** (default 5 GB is far too small: data 3 GB + parquet/features ~15 GB)
- IAM role: Create a new role → defaults (grants S3 access)
- Optional but recommended: a Lifecycle configuration with the AWS "auto-stop idle notebook" script (1 h idle)
- Create, wait for InService, Open JupyterLab → File → New → Terminal

## 3. Bootstrap and run
```
cd ~/SageMaker && aws s3 cp s3://<bucket>/code/business_entity_resolution/aws/bootstrap.sh . && bash bootstrap.sh <bucket>
cd ~/SageMaker/ber/code/business_entity_resolution/src
export BER_S3_BUCKET=<bucket>
nohup bash ../aws/run_all.sh > ~/SageMaker/ber/run.log 2>&1 &
tail -f ~/SageMaker/ber/run.log
```
`run_all.sh` is resumable: rerunning skips stages whose outputs exist. Delete the
corresponding file under `work/` to redo a stage (e.g. `rm work/model.txt` to retrain).

## 4. Stop the instance when done
Notebook instances → select → **Stop** (not Delete; the 100 GB volume keeps your `work/`
directory and costs ~$0.14/GB-month while stopped). A running `ml.g5.xlarge` left on
overnight costs ~$15–25; that is the only way to blow the budget.
