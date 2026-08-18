# Deployment Runbook — AWS SageMaker (M6)

This is the runbook to deploy the containerized model (`breast-cancer-api:latest`)
to a real cloud endpoint on **Amazon SageMaker**, using the bring-your-own-container
(BYOC) path. It is documented rather than executed here to avoid cloud charges — a
running endpoint bills by the hour.

> **Cost warning:** a real-time SageMaker endpoint runs (and bills) until you delete
> it. Always run the teardown step (§7) when finished. Prefer **Serverless Inference**
> (§8) for a demo — you pay per request, not per hour.

---

## 0. Prerequisites
- AWS account + **AWS CLI** configured (`aws configure`)
- An **IAM execution role** for SageMaker with `AmazonSageMakerFullAccess` and ECR
  pull permissions (note its ARN — referred to below as `$ROLE_ARN`)
- Docker running locally (the image already built in M2)

```bash
export AWS_REGION=us-east-1
export ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
export ECR_REPO=breast-cancer-api
export IMAGE_URI=$ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_REPO:latest
export ROLE_ARN=arn:aws:iam::$ACCOUNT_ID:role/<your-sagemaker-execution-role>
```

---

## 1. Make the container SageMaker-compatible (one small code change)

SageMaker's contract: the container must listen on **port 8080** and expose
- `GET /ping` → return **200** when healthy
- `POST /invocations` → run inference

Our FastAPI app uses `/health` and `/predict` on 8000, so add thin aliases in
`src/serve.py` (keeps the existing endpoints working too):

```python
@app.get("/ping")
def ping():
    return {"status": "ok"}          # SageMaker health check

@app.post("/invocations")
def invocations(req: PredictRequest):
    return predict(req)              # reuse the existing handler
```

And expose 8080 in the `Dockerfile` (or override the CMD at deploy time):

```dockerfile
EXPOSE 8080
CMD ["uvicorn", "src.serve:app", "--host", "0.0.0.0", "--port", "8080"]
```

Rebuild after the change: `docker build -t breast-cancer-api:latest .`

---

## 2. Create the ECR repo and push the image

```bash
aws ecr create-repository --repository-name $ECR_REPO --region $AWS_REGION

aws ecr get-login-password --region $AWS_REGION \
  | docker login --username AWS --password-stdin \
    $ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com

docker tag breast-cancer-api:latest $IMAGE_URI
docker push $IMAGE_URI
```

---

## 3. Create the SageMaker Model

```bash
aws sagemaker create-model \
  --model-name breast-cancer-model \
  --primary-container Image=$IMAGE_URI \
  --execution-role-arn $ROLE_ARN \
  --region $AWS_REGION
```

The model artifact is already baked into the image (`MODEL_PATH`), so no separate
`ModelDataUrl` / S3 upload is needed.

---

## 4. Create the Endpoint Configuration

```bash
aws sagemaker create-endpoint-config \
  --endpoint-config-name breast-cancer-config \
  --production-variants \
    VariantName=AllTraffic,ModelName=breast-cancer-model,\
InstanceType=ml.t2.medium,InitialInstanceCount=1 \
  --region $AWS_REGION
```

---

## 5. Create the Endpoint (this is what bills)

```bash
aws sagemaker create-endpoint \
  --endpoint-name breast-cancer-endpoint \
  --endpoint-config-name breast-cancer-config \
  --region $AWS_REGION

# Wait until InService (a few minutes):
aws sagemaker wait endpoint-in-service \
  --endpoint-name breast-cancer-endpoint --region $AWS_REGION
```

---

## 6. Invoke the endpoint

```python
import json, boto3
rt = boto3.client("sagemaker-runtime", region_name="us-east-1")
payload = {"features": [ ...30 floats... ]}   # GET /example gives a sample
resp = rt.invoke_endpoint(
    EndpointName="breast-cancer-endpoint",
    ContentType="application/json",
    Body=json.dumps(payload),
)
print(json.loads(resp["Body"].read()))
```

---

## 7. Teardown (do this to stop charges)

```bash
aws sagemaker delete-endpoint        --endpoint-name breast-cancer-endpoint --region $AWS_REGION
aws sagemaker delete-endpoint-config --endpoint-config-name breast-cancer-config --region $AWS_REGION
aws sagemaker delete-model           --model-name breast-cancer-model --region $AWS_REGION
```

---

## 8. Cheaper / alternative paths

- **Serverless Inference** — same model/endpoint-config, but set a
  `ServerlessConfig` (memory + max concurrency) instead of an instance type. Pay per
  request, scales to zero. Best for a portfolio demo.
- **Batch Transform** — for scoring a file of records offline, no standing endpoint.
- **Other clouds** — the same image deploys to **GCP Vertex AI** (custom container,
  health route `/ping`, predict route `/predict`) or **Azure ML** with minor changes.

---

## CI/CD hook (optional next step)

A `deploy` job in `.github/workflows/ci.yml` could, on a tagged release, log in to
ECR, push the image, and `update-endpoint` — giving full train→test→build→deploy
automation. Gate it on the test job passing.
