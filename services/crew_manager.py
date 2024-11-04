from typing import Dict
from crewai import Crew, Task
from agents.transcript_agent import TranscriptAgent
from agents.pika_agent import PikaWebAgent
from agents.youtube_agent import YouTubeAgent

class CrewManager:
    def __init__(self):
        self.transcript_agent = TranscriptAgent.create()
        self.video_agent = PikaWebAgent.create()
        self.youtube_agent = YouTubeAgent.create()

    def create_transcript(self, prompt: str) -> Dict:
        task = Task(
            description=f"Create a video transcript based on: {prompt}",
            agent=self.transcript_agent
        )
        
        crew = Crew(
            agents=[self.transcript_agent],
            tasks=[task]
        )
        
        result = crew.kickoff()
        return {"transcript": result}

    def generate_video(self, transcript: str) -> Dict:
        task = Task(
            description=f"Generate video from transcript: {transcript}",
            agent=self.video_agent
        )
        
        crew = Crew(
            agents=[self.video_agent],
            tasks=[task]
        )
        
        result = crew.kickoff()
        return {"videoUrl": result}

    def upload_to_youtube(self, video_url: str, status: str = 'private') -> Dict:
        task = Task(
            description=f"Upload video to YouTube: {video_url} with status: {status}",
            agent=self.youtube_agent
        )
        
        crew = Crew(
            agents=[self.youtube_agent],
            tasks=[task]
        )
        
        result = crew.kickoff()
        return {"status": "success", "youtubeUrl": result}