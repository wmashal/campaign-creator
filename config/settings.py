class Config:
    DEBUG = True
    API_PREFIX = '/api'
    CORS_ORIGINS = ['http://localhost:3000']  # React frontend URL
    
    # Add other configuration settings as needed
    YOUTUBE_API_KEY = 'your-youtube-api-key'
    VIDEO_STORAGE_PATH = 'videos'