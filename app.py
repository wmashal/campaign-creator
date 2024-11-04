# app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from crewai import Agent, Task, Crew
import os
from agents.transcript_agent import TranscriptAgent
import logging
import traceback
import sys
from agents.pika_agent import PikaWebAgent
from agents.runway_agent import RunwayAgent

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
app = Flask(__name__)
CORS(app)
transcript_agent = TranscriptAgent(os.getenv('OPENAI_API_KEY'))

pika_agent = PikaWebAgent(
    openai_api_key=os.getenv('OPENAI_API_KEY'),
    pika_api_key=os.getenv('PIKA_API_KEY')
)

runway_agent = RunwayAgent(
    openai_api_key=os.getenv('OPENAI_API_KEY'),
    runway_api_key=os.getenv('RUNWAY_API_KEY')
)

# Create agents
content_writer = Agent(
    role='Content Writer',
    goal='Create engaging video transcripts',
    backstory='Expert in creating compelling video content and scripts',
    verbose=True,
    allow_delegation=False
)

video_producer = Agent(
    role='Video Producer',
    goal='Generate high-quality videos from transcripts',
    backstory='Specialist in video production and AI video generation',
    verbose=True,
    allow_delegation=False
)

youtube_manager = Agent(
    role='YouTube Manager',
    goal='Handle YouTube video uploads and publishing',
    backstory='Expert in YouTube content management and optimization',
    verbose=True,
    allow_delegation=False
)

class CustomJSONEncoder(Flask.json_encoder):
    def default(self, obj):
        try:
            return str(obj)
        except:
            return super().default(obj)

app.json_encoder = CustomJSONEncoder

@app.route('/api/generate-transcript', methods=['POST'])
def generate_transcript():
    try:
        data = request.json
        logger.debug(f"Received request data: {data}")
        
        if not data or 'prompt' not in data:
            return jsonify({
                "status": "error",
                "message": "No prompt provided"
            }), 400
            
        prompt = data['prompt'].strip()
        if not prompt:
            return jsonify({
                "status": "error",
                "message": "Prompt cannot be empty"
            }), 400
            
        result = transcript_agent.generate_transcript(prompt)
        logger.debug(f"Generated transcript result: {result}")
        
        if result["status"] == "error":
            return jsonify(result), 500
            
        # Ensure the transcript is a string
        if result["transcript"] is not None:
            result["transcript"] = str(result["transcript"])
            
        return jsonify({
            "status": "success",
            "transcript": result["transcript"],
            "message": "Transcript generated successfully"
        })
        
    except Exception as e:
        error_details = traceback.format_exc()
        logger.error(f"Error in generate_transcript: {error_details}")
        return jsonify({
            "status": "error",
            "message": str(e),
            "details": error_details
        }), 500

@app.route('/api/generate-video', methods=['POST'])
def generate_video():
    try:
        data = request.json
        logger.debug(f"Received video generation request: {data}")
        
        # Ensure all required fields are present
        if not data:
            return jsonify({
                "status": "error",
                "message": "No data provided"
            }), 400
            
        # Prepare the request data
        video_request = {
            "promptText": data.get('promptText', ''),
            "model": data.get('model', '1.5'),
            "pikaffect": data.get('pikaffect'),
            "options": {
                "aspectRatio": "5:2",
                "frameRate": 24,
                "camera": {
                    "pan": "none",
                    "tilt": "none",
                    "rotate": "none",
                    "zoom": "none"
                },
                "parameters": {
                    "motion": 1,
                    "guidanceScale": 12,
                    "negativePrompt": "",
                    "seed": None
                },
                "extend": False
            }
        }
        
        # Update options if provided
        if 'options' in data:
            video_request['options'].update(data['options'])
        
        # Start generation
        result = pika_agent.generate_video(video_request)
        logger.debug(f"Generation initiation result: {result}")
        
        if result.get("status") == "error":
            return jsonify(result), 500
            
        return jsonify({
            "status": "success",
            "job_id": result["job_id"],
            "message": result["message"],
            "details": result.get("details", {})
        })
        
    except Exception as e:
        error_details = traceback.format_exc()
        logger.error(f"Error in generate_video: {error_details}")
        return jsonify({
            "status": "error",
            "message": str(e),
            "details": error_details
        }), 500

@app.route('/api/runway/generate-video', methods=['POST'])
def generate_video_runway():
    """Endpoint for Runway video generation"""
    try:
        data = request.json
        logger.debug(f"Received Runway generation request: {data}")
        
        result = runway_agent.generate_video(data)
        logger.debug(f"Runway generation result: {result}")
        
        if result.get("status") == "error":
            return jsonify(result), 500
            
        return jsonify(result)
        
    except Exception as e:
        error_details = traceback.format_exc()
        logger.error(f"Error in Runway generation: {error_details}")
        return jsonify({
            "status": "error",
            "message": str(e),
            "details": error_details
        }), 500
        
@app.route('/api/video-status/<job_id>', methods=['GET'])
def get_video_status(job_id):
    try:
        status = pika_agent.check_job_status(job_id)
        return jsonify(status)
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route('/api/reprompt-video', methods=['POST'])
def reprompt_video():
    try:
        data = request.json
        logger.debug(f"Received reprompt request: {data}")
        
        if not data or 'promptText' not in data or 'video' not in data:
            return jsonify({
                "status": "error",
                "message": "Missing required parameters"
            }), 400
        
        result = pika_agent.reprompt_video(
            data['promptText'],
            data['video'],
            data['options']
        )
        
        logger.debug(f"Reprompt result: {result}")
        
        if result.get("status") == "error":
            return jsonify(result), 500
            
        return jsonify(result)
        
    except Exception as e:
        error_details = traceback.format_exc()
        logger.error(f"Error in reprompt-video: {error_details}")
        return jsonify({
            "status": "error",
            "message": str(e),
            "details": error_details
        }), 500
        
@app.route('/api/upload-youtube', methods=['POST'])
def upload_youtube():
    try:
        data = request.json
        task = Task(
            description=f"Upload video to YouTube: {data.get('videoUrl', '')} with status: {data.get('status', 'private')}",
            agent=youtube_manager
        )
        
        crew = Crew(
            agents=[youtube_manager],
            tasks=[task]
        )
        
        result = crew.kickoff()
        return jsonify({
            "status": "success",
            "youtubeUrl": "https://youtube.com/watch?v=test123"  # Placeholder URL
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

if __name__ == '__main__':
    required_vars = ['OPENAI_API_KEY', 'PIKA_API_KEY','RUNWAY_API_KEY']
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    if missing_vars:
        logger.error(f"Missing required environment variables: {', '.join(missing_vars)}")
        sys.exit(1)
        
    app.run(debug=True)