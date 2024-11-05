import requests
import json
import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)

class RunwayAgent:
    def __init__(self, openai_api_key: str, runway_api_key: str):
        self.runway_api_key = runway_api_key
        self.base_url = "https://api.useapi.net"
        self.headers = {
            "Authorization": f"Bearer {runway_api_key}",
            "Content-Type": "application/json"
        }
        
    def get_assets(self, media_type: str = 'image', offset: int = 0, limit: int = 50) -> Dict:
        """Get list of assets from Runway"""
        try:
            headers = {
                "Authorization": f"Bearer {self.runway_api_key}",
                "Content-Type": "application/json"
            }
            
            # Build URL with required parameters
            url = f"{self.base_url}/v1/runwayml/assets/"  # Note the trailing slash
            
            # Include all parameters in the params dictionary
            params = {
                "mediaType": media_type,
                "offset": str(offset),
                "limit": str(limit)
            }

            logger.debug(f"Getting assets from URL: {url}")
            logger.debug(f"Headers: {headers}")
            logger.debug(f"Params: {params}")
            
            # Make the request
            response = requests.get(
                url,
                headers=headers,
                params=params
            )
            
            # Log the actual URL that was called
            logger.debug(f"Full URL with params: {response.url}")
            logger.debug(f"Response status code: {response.status_code}")
            logger.debug(f"Response headers: {dict(response.headers)}")
            
            # Check if we got a successful response
            if response.status_code == 200:
                try:
                    result = response.json()
                    logger.debug(f"Parsed response: {result}")
                    return {
                        "status": "success",
                        "assets": result
                    }
                except json.JSONDecodeError as je:
                    logger.error(f"Failed to parse JSON response: {response.text}")
                    return {
                        "status": "error",
                        "message": f"Invalid JSON response: {str(je)}"
                    }
            else:
                logger.error(f"API returned error status: {response.status_code}")
                logger.error(f"Response content: {response.text}")
                return {
                    "status": "error",
                    "message": f"API returned status code {response.status_code}"
                }
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed: {str(e)}")
            return {
                "status": "error",
                "message": f"Request failed: {str(e)}"
            }
        except Exception as e:
            logger.error(f"Error getting assets: {str(e)}")
            return {
                "status": "error",
                "message": f"Error getting assets: {str(e)}"
            }

    def upload_asset(self, file_data: bytes, file_name: str, content_type: str) -> Dict:
        """Upload an image asset to Runway"""
        try:
            files = {
                'file': (file_name, file_data, content_type)
            }
            
            params = {
                'name': file_name
            }
            
            # Updated URL with correct endpoint
            url = f"{self.base_url}/v1/runwayml/assets"
            logger.debug(f"Uploading asset to URL: {url}")
            
            response = requests.post(
                url,
                headers={"Authorization": f"Bearer {self.runway_api_key}"},
                params=params,
                files=files
            )
            
            logger.debug(f"Upload response status: {response.status_code}")
            logger.debug(f"Upload response: {response.text}")
            
            response.raise_for_status()
            result = response.json()
            
            return {
                "status": "success",
                "asset_id": result.get("assetId"),
                "url": result.get("url"),
                "details": result
            }
        
        except requests.exceptions.HTTPError as e:
            logger.error(f"HTTP Error during upload: {e.response.text if e.response else str(e)}")
            return {
                "status": "error",
                "message": f"Upload failed: {str(e)}"
            }
        except Exception as e:
            logger.error(f"Error uploading asset: {str(e)}")
            return {
                "status": "error",
                "message": f"Error uploading asset: {str(e)}"
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
            for param in ['firstImage_assetId', 'lastImage_assetId', 'seed']:
                if data.get(param):
                    payload[param] = data[param]

            # Add motion controls if present
            for param in ['horizontal', 'vertical', 'roll', 'zoom', 'pan', 'tilt']:
                if data.get(param) is not None:
                    payload[param] = max(-10, min(10, data[param]))

            logger.debug(f"Sending generation request to Runway with payload: {payload}")
            
            response = requests.post(
                f"{self.base_url}/gen3turbo/create",
                headers=self.headers,
                json=payload
            )
            
            response.raise_for_status()
            result = response.json()
            logger.debug(f"Runway API Response: {result}")
            
            task_id = result.get('taskId')
            if not task_id:
                raise Exception("No taskId received from Runway API")
            
            return {
                "status": "pending",
                "job_id": task_id,
                "message": "Video generation started",
                "progress": 0
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
            
            status_map = {
                'PENDING': 'pending',
                'PROCESSING': 'processing',
                'SUCCEEDED': 'completed',
                'FAILED': 'failed'
            }
            
            runway_status = result.get('status', 'PENDING')
            mapped_status = status_map.get(runway_status, runway_status.lower())
            
            progress = 0
            if result.get('progressRatio'):
                progress = int(float(result['progressRatio']) * 100)
            elif mapped_status == 'completed':
                progress = 100
                
            status_response = {
                "status": mapped_status,
                "job_id": task_id,
                "progress": progress,
                "message": result.get('progressText', 'Processing video')
            }
            
            if mapped_status == 'completed':
                artifacts = result.get('artifacts', [])
                if artifacts:
                    video_artifact = next(
                        (a for a in artifacts if a.get('metadata', {}).get('frameRate')), 
                        None
                    )
                    if video_artifact:
                        status_response.update({
                            "video_url": video_artifact.get('url'),
                            "metadata": {
                                "duration": video_artifact.get('metadata', {}).get('duration'),
                                "dimensions": video_artifact.get('metadata', {}).get('dimensions'),
                                "frame_rate": video_artifact.get('metadata', {}).get('frameRate')
                            }
                        })
            elif mapped_status == 'failed':
                status_response.update({
                    "message": f"Generation failed: {result.get('error', 'Unknown error')}"
                })
            
            return status_response
            
        except Exception as e:
            logger.error(f"Error checking status: {str(e)}")
            return {
                "status": "error",
                "job_id": task_id,
                "message": f"Error checking status: {str(e)}"
            }