#!/usr/bin/env bash
# One-time setup of a machine for this pipeline: works on a SageMaker Notebook Instance
# (JupyterLab terminal) and on a plain EC2 instance (Ubuntu 22.04/24.04 or Amazon Linux 2023).
#
#   bash bootstrap.sh <s3-bucket>
#
# Assumes the challenge TSVs were uploaded to   s3://<bucket>/dataset/{train,test}/*.tsv
# the validator to                              s3://<bucket>/utils/validate_submission.py
# and this repo (code/business_entity_resolution) to  s3://<bucket>/code/business_entity_resolution
set -euo pipefail
BUCKET="${1:?usage: bootstrap.sh <s3-bucket>}"

if [ -d /home/ec2-user/SageMaker ]; then
  ROOT=/home/ec2-user/SageMaker/ber          # SageMaker notebook: the persistent EBS volume
  ON_SAGEMAKER=1
else
  ROOT="$HOME/ber"                           # EC2
  ON_SAGEMAKER=0
fi
mkdir -p "$ROOT"/{student_resource/dataset,student_resource/utils,work,output,code}
cd "$ROOT"

# 0. system packages + AWS CLI on EC2 (SageMaker images already have them)
if [ "$ON_SAGEMAKER" = 0 ]; then
  if command -v apt-get >/dev/null; then
    sudo apt-get update -qq && sudo apt-get install -y -qq python3-venv python3-pip unzip tmux htop
  elif command -v dnf >/dev/null; then
    sudo dnf install -y -q python3-pip unzip tmux htop
  fi
  if ! command -v aws >/dev/null; then
    curl -sS "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o /tmp/awscliv2.zip
    unzip -q -o /tmp/awscliv2.zip -d /tmp && sudo /tmp/aws/install
  fi
fi

# 1. code + data.  If the laptop uploaded its normalised intermediates (s3://<bucket>/work),
#    pull them too: a GPU box then only runs the embedding steps, never the full pipeline.
aws s3 sync "s3://$BUCKET/code/business_entity_resolution" code/business_entity_resolution
aws s3 sync "s3://$BUCKET/dataset" student_resource/dataset
aws s3 cp   "s3://$BUCKET/utils/validate_submission.py" student_resource/utils/validate_submission.py || true
aws s3 sync "s3://$BUCKET/work" work || true

# 2. python environment
if [ "$ON_SAGEMAKER" = 1 ]; then
  source activate python3 2>/dev/null || true
  PIP="pip"
else
  python3 -m venv "$ROOT/venv"
  PIP="$ROOT/venv/bin/pip"
  echo "source $ROOT/venv/bin/activate" >> ~/.bashrc
fi
$PIP install -q --upgrade pip
$PIP install -q -r code/business_entity_resolution/requirements.txt
$PIP install -q sentence-transformers faiss-cpu      # embedding experiment; harmless on CPU boxes

# 3. environment variables used by ber.config
cat >> ~/.bashrc <<EOF
export BER_ROOT=$ROOT
export BER_DATA_DIR=$ROOT/student_resource/dataset
export BER_WORK_DIR=$ROOT/work
export BER_OUTPUT_DIR=$ROOT/output
export BER_S3_BUCKET=$BUCKET
EOF

echo
echo "Setup done. Open a NEW shell (or: source ~/.bashrc), then:"
echo "  cd $ROOT/code/business_entity_resolution/src"
echo "  nohup bash ../aws/run_all.sh > $ROOT/run.log 2>&1 &"
echo "  tail -f $ROOT/run.log"
