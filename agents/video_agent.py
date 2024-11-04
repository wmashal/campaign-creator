from crewai import Agent, Task, Crew
from langchain_openai import ChatOpenAI
import requests
import json
import os
import logging
from typing import Dict, Optional
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class TextJustification(str, Enum):
    LEFT = "LEFT"
    CENTER = "CENTER"
    RIGHT = "RIGHT"

class VerticalAlignment(str, Enum):
    TOP = "TOP"
    MIDDLE = "MIDDLE"
    BOTTOM = "BOTTOM"

class BackgroundStyleType(str, Enum):
    RECT = "RECT"
    WRAPPED = "WRAPPED"

@dataclass
class Color:
    red: int
    green: int
    blue: int

    def to_dict(self):
        return {
            "red": self.red,
            "green": self.green,
            "blue": self.blue
        }

@dataclass
class AspectRatio:
    width: float
    height: float

    def to_dict(self):
        return {
            "width": self.width,
            "height": self.height
        }

class VideoAgent:
    def __init__(self, openai_api_key: str, videogen_api_key: str):
        self.videogen_api_key = videogen_api_key
        self.api_url = "https://ext.videogen.io/v1/script-to-video"
        
        self.llm = ChatOpenAI(
            temperature=0.7,
            model="gpt-4",
            api_key=openai_api_key
        )
        
        self.agent = Agent(
            role='Video Script Enhancer',
            goal='Convert transcripts into VGML format for video generation',
            backstory="""You are an expert in converting scripts into VideoGen 
            Markup Language (VGML) format, optimizing them for video generation.""",
            verbose=True,
            allow_delegation=False,
            llm=self.llm
        )

    def convert_to_vgml(self, transcript: str) -> str:
        """Convert transcript to VGML format"""
        try:
            task = Task(
                description=f"""Convert this transcript into VGML format:
                {transcript}
                
                Follow these rules:
                1. Add [SCENE] markers for each scene
                2. Add [TEXT] markers for captions
                3. Add [TRANSITION] markers between scenes
                4. Include timing information
                5. Specify camera movements
                
                Example VGML format:
                [SCENE duration="5"]
                [CAMERA move="pan-right"]
                [TEXT align="center"]Main content here[/TEXT]
                [/SCENE]
                """,
                expected_output="VGML formatted script",
                agent=self.agent
            )
            
            crew = Crew(
                agents=[self.agent],
                tasks=[task]
            )
            
            vgml_script = str(crew.kickoff())
            logger.debug(f"Generated VGML: {vgml_script}")
            return vgml_script
            
        except Exception as e:
            logger.error(f"Error converting to VGML: {str(e)}")
            raise

    async def generate_video(self, transcript: str, 
                           voice: str = "Matilda",
                           options: Optional[Dict] = None) -> Dict[str, str]:
        """Generate video using VideoGen API"""
        try:
            # Convert transcript to VGML
            vgml_script = self.convert_to_vgml(transcript)
            
            # Prepare request payload
            payload = {
                "script": vgml_script,
                "voice": voice,
                "voiceVolume": 1.0,
                "musicVolume": 0.15,
                "aspectRatio": {
                    "width": 16,
                    "height": 9
                },
                "minDimensionPixels": 1080,
                "captionFontSize": 75,
                "captionFontWeight": 700,
                "captionTextColor": Color(255, 255, 255).to_dict(),  # White
                "captionTextJustification": "CENTER",
                "captionVerticalAlignment": "BOTTOM",
                "captionBackgroundStyleType": "WRAPPED",
                "captionBackgroundColor": Color(0, 0, 0).to_dict(),  # Black
                "captionBackgroundOpacity": 0.5,
            }
            
            # Update with any custom options
            if options:
                payload.update(options)
            
            headers = {
                "Authorization": f"Bearer {self.videogen_api_key}",
                "Content-Type": "application/json"
            }
            
            logger.debug(f"Sending request to VideoGen API: {payload}")
            
            # Make API request
            response = requests.post(
                self.api_url,
                headers=headers,
                json=payload
            )
            response.raise_for_status()
            
            result = response.json()
            logger.debug(f"Received response from VideoGen API: {result}")
            
            return {
                "status": "success",
                "video_id": result.get("apiFileId"),
                "video_url": result.get("apiFileSignedUrl"),
                "message": "Video generation initiated successfully"
            }
            
        except Exception as e:
            logger.error(f"Error generating video: {str(e)}")
            return {
                "status": "error",
                "message": f"Error generating video: {str(e)}"
            }

    def check_video_status(self, video_id: str) -> Dict[str, str]:
        """Check status of video generation"""
        try:
            headers = {
                "Authorization": f"Bearer {self.videogen_api_key}",
                "Content-Type": "application/json"
            }
            
            response = requests.get(
                f"https://ext.videogen.io/v1/status/{video_id}",
                headers=headers
            )
            response.raise_for_status()
            
            return {
                "status": "success",
                "details": response.json()
            }
            
        except Exception as e:
            logger.error(f"Error checking video status: {str(e)}")
            return {
                "status": "error",
                "message": f"Error checking status: {str(e)}"
            }