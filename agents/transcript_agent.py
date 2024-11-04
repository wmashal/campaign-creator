from crewai import Agent, Task, Crew
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from typing import Dict
import logging
import os
import traceback
import json

logger = logging.getLogger(__name__)

class TranscriptAgent:
    def __init__(self, openai_api_key: str):
        if not openai_api_key:
            raise ValueError("OpenAI API key is required")
            
        self.llm = ChatOpenAI(
            temperature=0.7,
            model="gpt-4",
            api_key=openai_api_key
        )
        
        self.agent = Agent(
            role='Content Writer',
            goal='Create engaging video transcripts based on campaign requirements',
            backstory="""You are an expert content creator specializing in creating 
            engaging video scripts. You understand how to structure content for 
            video format, including pacing, tone, and engagement hooks.""",
            verbose=True,
            allow_delegation=False,
            llm=self.llm
        )
        
        self.prompt_template = PromptTemplate(
            input_variables=["campaign_details"],
            template="""Create a professional video script based on the following campaign details:
            {campaign_details}
            
            Consider the following in your script:
            1. Start with a strong hook to grab attention
            2. Include clear call-to-actions
            3. Maintain an engaging pace
            4. Use conversational language
            5. Include timing markers [00:00] for each section
            6. Keep the total length between 2-3 minutes
            
            Format the script with:
            - Speaking parts in regular text
            
            Script:"""
        )

    def generate_transcript(self, campaign_details: str) -> Dict[str, str]:
        try:
            logger.debug(f"Generating transcript for: {campaign_details}")
            
            task = Task(
                description=self.prompt_template.format(
                    campaign_details=campaign_details
                ),
                expected_output="""A video script formatted with timing markers, 
                visual descriptions in brackets, and speaking parts in regular text. 
                The script should be 2-3 minutes long.""",
                agent=self.agent
            )
            
            crew = Crew(
                agents=[self.agent],
                tasks=[task]
            )
            
            result = crew.kickoff()
            # Convert CrewOutput to string
            transcript = str(result)
            logger.debug(f"Generated transcript: {transcript}")
            
            if not transcript:
                raise ValueError("Generated transcript is empty")
                
            return {
                "status": "success",
                "transcript": transcript,
                "message": "Transcript generated successfully"
            }
            
        except Exception as e:
            error_details = traceback.format_exc()
            logger.error(f"Error generating transcript: {error_details}")
            return {
                "status": "error",
                "transcript": None,
                "message": f"Error generating transcript: {str(e)}"
            }

    def regenerate_transcript(self, campaign_details: str) -> Dict[str, str]:
        try:
            self.llm.temperature = 0.9
            result = self.generate_transcript(campaign_details)
            self.llm.temperature = 0.7
            return result
        except Exception as e:
            logger.error(f"Error regenerating transcript: {str(e)}")
            return {
                "status": "error",
                "transcript": None,
                "message": f"Error regenerating transcript: {str(e)}"
            }