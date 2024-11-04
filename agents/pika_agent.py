import requests
import json
import logging
import time
from typing import Dict, Optional
from crewai import Agent, Task, Crew
from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)

class PikaWebAgent:
    def __init__(self, openai_api_key: str, pika_api_key: str):
        self.pika_api_key = pika_api_key
        self.base_url = "https://api.pikapikapika.io/web"
        self.headers = {
            "Authorization": f"Bearer {pika_api_key}",
            "Content-Type": "application/json"
        }
        
        self.llm = ChatOpenAI(
            temperature=0.7,
            model="gpt-4",
            api_key=openai_api_key
        )
        
        self.agent = Agent(
            role='Pika Prompt Engineer',
            goal='Create optimized prompts for Pika video generation',
            backstory="""You are an expert in crafting perfect prompts for 
            Pika's text-to-video generation.""",
            verbose=True,
            allow_delegation=False,
            llm=self.llm
        )

    def optimize_prompt(self, transcript: str) -> str:
        """Convert transcript into optimized prompt"""
        try:
            task = Task(
                description=f"""Create a concise visual prompt for video generation from this transcript:
                {transcript}
                
                Focus on:
                1. Visual elements and scenes
                2. Actions and movements
                3. Style and atmosphere
                Keep it under 300 characters.""",
                expected_output="Concise visual prompt",
                agent=self.agent
            )
            
            crew = Crew(
                agents=[self.agent],
                tasks=[task]
            )
            
            prompt = str(crew.kickoff()).strip()
            return prompt[:300]  # Ensure within character limit
            
        except Exception as e:
            logger.error(f"Error optimizing prompt: {str(e)}")
            return transcript[:300]

    def generate_video(self, data: Dict) -> Dict:
        """Generate video using Pika Web API"""
        try:
            # Extract main parameters from data
            transcript = data.get('promptText', '')
            model_version = data.get('model', '1.5')
            pika_effect = data.get('pikaffect')
            user_options = data.get('options', {})
            
            # Format prompt with effect if provided
            prompt_text = f"{pika_effect} {transcript}" if pika_effect else transcript
            
            # Construct the API payload
            payload = {
                "promptText": prompt_text,
                "model": model_version,
                "pikaffect": pika_effect,
                "options": {
                    "aspectRatio": user_options.get('aspectRatio', '5:2'),
                    "frameRate": user_options.get('frameRate', 24),
                    "camera": user_options.get('camera', {
                        "pan": "none",
                        "tilt": "none",
                        "rotate": "none",
                        "zoom": "none"
                    }),
                    "parameters": {
                        "motion": user_options.get('parameters', {}).get('motion', 1),
                        "guidanceScale": user_options.get('parameters', {}).get('guidanceScale', 12),
                        "negativePrompt": user_options.get('parameters', {}).get('negativePrompt', ''),
                        "seed": user_options.get('parameters', {}).get('seed')
                    },
                    "extend": user_options.get('extend', False)
                }
            }
            
            logger.debug(f"Sending generation request with payload: {payload}")
            
            # Make the API request
            response = requests.post(
                f"{self.base_url}/generate",
                headers=self.headers,
                json=payload
            )
            
            if response.status_code != 200:
                logger.error(f"API Error: {response.status_code} - {response.text}")
                return {
                    "status": "error",
                    "message": f"API Error: {response.text}"
                }
            
            result = response.json()
            logger.debug(f"API Response: {result}")
            
            # Extract job ID from response
            if isinstance(result, dict):
                job_id = (
                    result.get('job', {}).get('id') or 
                    result.get('video', {}).get('jobId') or 
                    result.get('jobId')
                )
                
                if job_id:
                    return {
                        "status": "pending",
                        "job_id": job_id,
                        "message": "Video generation started",
                        "details": result
                    }
            
            raise Exception(f"Invalid API response format: {result}")
            
        except Exception as e:
            logger.error(f"Error starting video generation: {str(e)}")
            return {
                "status": "error",
                "message": f"Error starting video generation: {str(e)}"
            }

    def check_job_status(self, job_id: str) -> Dict:
        """Check status of a video generation job"""
        try:
            response = requests.get(
                f"{self.base_url}/jobs/{job_id}",
                headers=self.headers
            )
            
            if response.status_code != 200:
                logger.error(f"Status check error: {response.status_code} - {response.text}")
                return {
                    "status": "error",
                    "message": f"Status check failed: {response.text}"
                }
            
            result = response.json()
            logger.debug(f"Status check response: {result}")
            
            # Extract job status
            job_data = result.get('job', {})
            videos_data = result.get('videos', [{}])[0] if result.get('videos') else {}
            
            job_status = job_data.get('status') or videos_data.get('status')
            
            status_data = {
                "status": job_status or "unknown",
                "job_id": job_id,
                "last_checked": time.strftime("%H:%M:%S"),
                "next_check": "30 seconds"
            }
            
            # Handle different status cases
            if job_status == "finished":
                video_url = videos_data.get('resultUrl') or videos_data.get('sharingUrl')
                poster_url = videos_data.get('videoPoster')
                status_data.update({
                    "status": "completed",
                    "video_url": video_url,
                    "poster_url": poster_url,
                    "progress": 100,
                    "message": "Video generation completed"
                })
            elif job_status == "failed":
                status_data.update({
                    "message": f"Generation failed: {job_data.get('error', 'Unknown error')}"
                })
            elif job_status in ["queued", "processing"]:
                status_data.update({
                    "progress": videos_data.get('progress', 0),
                    "message": "Video generation in progress",
                    "estimated_time": "Processing time varies based on queue and video complexity"
                })
            
            return status_data
                
        except Exception as e:
            logger.error(f"Error checking job status: {str(e)}")
            return {
                "status": "error",
                "job_id": job_id,
                "message": f"Error checking status: {str(e)}"
            }
            
def reprompt_video(self, new_prompt: str, previous_video_url: str, previous_options: Dict) -> Dict:
    """Reprompt a video with a new prompt while keeping previous options"""
    try:
        payload = {
            "promptText": new_prompt,
            "video": previous_video_url,
            "options": previous_options
        }
        
        logger.debug(f"Sending reprompt request with payload: {payload}")
        
        response = requests.post(
            f"{self.base_url}/generate",
            headers=self.headers,
            json=payload
        )
        
        if response.status_code != 200:
            logger.error(f"API Error: {response.status_code} - {response.text}")
            return {
                "status": "error",
                "message": f"API Error: {response.text}"
            }
        
        result = response.json()
        logger.debug(f"Reprompt API Response: {result}")
        
        # Extract job ID from response
        if isinstance(result, dict):
            job_id = result.get('job', {}).get('id') or result.get('video', {}).get('jobId')
            
            if job_id:
                return {
                    "status": "pending",
                    "job_id": job_id,
                    "message": "Video regeneration started",
                    "details": result
                }
        
        raise Exception(f"Invalid API response format: {result}")
        
    except Exception as e:
        logger.error(f"Error in reprompt: {str(e)}")
        return {
            "status": "error",
            "message": f"Error regenerating video: {str(e)}"
        }