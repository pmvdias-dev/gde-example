import boto3
from botocore.exceptions import ClientError, NoCredentialsError

AWS_REGION = "us-east-1"
MODEL_ID = "global.anthropic.claude-opus-4-6-v1"

try:
    session = boto3.Session(region_name=AWS_REGION)

    identity = session.client("sts").get_caller_identity()
    print("Account:", identity["Account"])
    print("ARN:", identity["Arn"])

    bedrock = session.client("bedrock-runtime")

    response = bedrock.converse(
        modelId=MODEL_ID,
        messages=[
            {
                "role": "user",
                "content": [{"text": "Hello! Explain Amazon Bedrock briefly."}],
            }
        ],
        inferenceConfig={"maxTokens": 300, "temperature": 0.5},
    )

    for block in response["output"]["message"]["content"]:
        if "text" in block:
            print(block["text"])

    print("Usage:", response.get("usage", {}))

except NoCredentialsError:
    print("AWS credentials not found. Run: aws configure")

except ClientError as error:
    print("Error:", error.response["Error"]["Code"])
    print("Message:", error.response["Error"]["Message"])