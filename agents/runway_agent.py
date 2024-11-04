import requests
import json
import logging
import time
from typing import Dict, Optional
from crewai import Agent, Task, Crew
from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)

class RunwayAgent:
    def __init__(self, openai_api_key: str, runway_api_key: str):
        self.runway_api_key = runway_api_key
        self.base_url = "https://api.useapi.net/v1/runwayml"
        self.headers = {
            "Authorization": f"Bearer {runway_api_key}",
            "Content-Type": "application/json"
        }

    def generate_video(self, data: Dict) -> Dict:
        """Generate video using Runway API"""
        try:
            # Extract main parameters and ensure correct format
            payload = {
                "text_prompt": data.get('text_prompt', ''),
                "aspect_ratio": data.get('aspect_ratio', 'landscape'),
                "seconds": data.get('seconds', 5),
                "exploreMode": data.get('exploreMode', False),
            }

            # Add optional parameters only if they are present
            optional_params = [
                'firstImage_assetId',
                'lastImage_assetId',
                'seed',
                'replyUrl',
                'replyRef',
                'maxJobs'
            ]
            
            for param in optional_params:
                if param in data and data[param] is not None:
                    payload[param] = data[param]

            # Add motion controls if present
            motion_params = ['horizontal', 'vertical', 'roll', 'zoom', 'pan', 'tilt']
            for param in motion_params:
                if param in data:
                    value = data[param]
                    # Ensure motion values are within -10 to 10 range
                    if value is not None:
                        payload[param] = max(-10, min(10, value))

            # Validate payload parameters
            if 'seed' in payload and not (1 <= payload['seed'] <= 4294967294):
                raise ValueError("Seed must be between 1 and 4294967294")
                
            if 'seconds' in payload and payload['seconds'] not in [5, 10]:
                payload['seconds'] = 5  # Default to 5 if invalid
                
            if 'maxJobs' in payload and not (1 <= payload['maxJobs'] <= 10):
                payload['maxJobs'] = 1  # Default to 1 if invalid

            logger.debug(f"Sending generation request to Runway with payload: {payload}")
            
            # Make the API request
            response = requests.post(
                f"{self.base_url}/gen3turbo/create",
                headers=self.headers,
                json=payload
            )
            
            response.raise_for_status()
            result = response.json()
            logger.debug(f"Runway API Response: {result}")
            
            # Extract required information from response
            task_id = result.get('taskId')
            if not task_id:
                raise Exception("No taskId received from Runway API")
            
            # Format response to match expected structure
            return {
                "status": "pending",
                "task_id": task_id,
                "details": {
                    "task_type": result.get('taskType'),
                    "created_at": result.get('createdAt'),
                    "updated_at": result.get('updatedAt'),
                    "options": result.get('options', {}),
                    "status": result.get('status'),
                    "progress_text": result.get('progressText'),
                    "progress_ratio": result.get('progressRatio'),
                    "estimated_start": result.get('estimatedTimeToStartSeconds')
                },
                "message": "Video generation started"
            }
            
        except requests.exceptions.HTTPError as e:
            error_message = str(e.response.text if e.response else e)
            logger.error(f"HTTP Error: {error_message}")
            return {
                "status": "error",
                "message": f"Runway API HTTP Error: {error_message}"
            }
        except ValueError as e:
            logger.error(f"Validation Error: {str(e)}")
            return {
                "status": "error",
                "message": str(e)
            }
        except Exception as e:
            logger.error(f"Error starting video generation: {str(e)}")
            return {
                "status": "error",
                "message": f"Error starting video generation: {str(e)}"
            }

    def check_status(self, task_id: str) -> Dict:
        """Check status of a Runway video generation task"""
        try:
            response = requests.get(
                f"{self.base_url}/tasks/{task_id}",
                headers=self.headers
            )
            
            response.raise_for_status()
            result = response.json()
            logger.debug(f"Runway status check response: {result}")
            
            # Extract essential information
            status = result.get('status', 'PENDING')
            
            status_data = {
                "status": status.lower(),
                "task_id": task_id,
                "created_at": result.get('createdAt'),
                "updated_at": result.get('updatedAt'),
                "task_type": result.get('taskType'),
                "progress_text": result.get('progressText'),
                "progress_ratio": result.get('progressRatio'),
                "estimated_start": result.get('estimatedTimeToStartSeconds')
            }
            
            # Calculate progress percentage
            if result.get('progressRatio') is not None:
                status_data["progress"] = int(float(result['progressRatio']) * 100)
            
            # Handle different status cases
            if status == 'SUCCEEDED':
                # Extract video URL from artifacts
                artifacts = result.get('artifacts', [])
                if artifacts:
                    video_artifact = next((
                        a for a in artifacts 
                        if a.get('metadata', {}).get('frameRate')
                    ), None)
                    
                    if video_artifact:
                        status_data.update({
                            "status": "completed",
                            "video_url": video_artifact.get('url'),
                            "metadata": {
                                "duration": video_artifact.get('metadata', {}).get('duration'),
                                "dimensions": video_artifact.get('metadata', {}).get('dimensions'),
                                "frame_rate": video_artifact.get('metadata', {}).get('frameRate'),
                                "file_size": video_artifact.get('fileSize')
                            }
                        })
            elif status == 'FAILED':
                status_data.update({
                    "status": "failed",
                    "error": result.get('error'),
                    "message": f"Generation failed: {result.get('error', 'Unknown error')}"
                })
            
            # Add generation options for reference
            status_data["options"] = result.get('options', {})
            
            return status_data
            
        except requests.exceptions.HTTPError as e:
            error_message = str(e.response.text if e.response else e)
            logger.error(f"HTTP Error checking status: {error_message}")
            return {
                "status": "error",
                "task_id": task_id,
                "message": f"Error checking status: {error_message}"
            }
        except Exception as e:
            logger.error(f"Error checking status: {str(e)}")
            return {
                "status": "error",
                "task_id": task_id,
                "message": f"Error checking status: {str(e)}"
            }