'''
Image Confidence Lambda Function
'''
# -*- coding: utf-8 -*-

import json
import boto3
import base64

s3 = boto3.client("s3")


def lambda_handler(event, context):
    """A function to serialize target data from S3"""

    # Get the s3 address from the Step Function event input
    key = event["s3_key"]
    bucket = event["s3_bucket"]

    # Download the data from s3 to /tmp/image.png
    s3.download_file(bucket, key, "/tmp/image.png")

    # We read the data from a file
    with open("/tmp/image.png", "rb") as f:
        image_data = base64.b64encode(f.read())

    # Pass the data back to the Step Function
    print("Event:", event.keys())
    return {
        "statusCode": 200,
        "body": {
            "image_data": image_data,
            "s3_bucket": bucket,
            "s3_key": key,
            "inferences": [],
        },
    }


'''
Image Classifier Lambda Function
'''
# -*- coding: utf-8 -*-

import json
import sagemaker
import base64
from sagemaker.serializers import IdentitySerializer
from sagemaker.predictor import Predictor

# Fill this in with the name of your deployed model
ENDPOINT = "image-classification-2023-01-07-16-06-22-115"


def handler(event, context):
    event = event["body"]
    # Decode the image data
    image = base64.b64decode(event["image_data"])

    # Instantiate a Predictor
    predictor = Predictor(ENDPOINT)

    # For this model the IdentitySerializer needs to be "image/png"
    predictor.serializer = IdentitySerializer("image/png")

    # Make a prediction:
    inferences = predictor.predict(image)

    # We return the data back to the Step Function
    event["inferences"] = json.loads(inferences.decode("utf-8"))
    return {
        "statusCode": 200,
        "body": {
            "image_data": event["image_data"],
            "s3_bucket": "sagemaker-us-east-1-781937109705",
            "s3_key": event["s3_key"],
            "inferences": event["inferences"],
        },
    }


'''
Image Confidence Lambda Function
'''
# -*- coding: utf-8 -*-

import json


THRESHOLD = 0.93


def handler(event, context):
    event = event["body"]
    # Grab the inferences from the event
    inferences = event["inferences"]

    # Check if any values in our inferences are above THRESHOLD
    meets_threshold = (
        True if (inferences[0] > THRESHOLD) | (inferences[1] > THRESHOLD) else False
    )

    # If our threshold is met, pass our data back out of the
    # Step Function, else, end the Step Function with an error
    if meets_threshold:
        return {
            "statusCode": 200,
            "body": {
                "image_data": event["image_data"],
                "s3_bucket": "sagemaker-us-east-1-781937109705",
                "s3_key": event["s3_key"],
                "inferences": event["inferences"],
            },
        }

    return {
        "statusCode": 400,
        "body": {"error_message": "THRESHOLD_CONFIDENCE_NOT_MET"},
    }
