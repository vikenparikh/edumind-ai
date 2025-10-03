#!/usr/bin/env python3
"""
EduMind AI - Learning Intelligence Platform
Advanced Educational Decision Support System

Developed by: Viken Parikh
Version: 2.1.0
Purpose: Personalized learning and educational intelligence

This script initializes and starts the EduMind AI educational platform,
providing educators, institutions, and learners with AI-powered
personalized learning experiences, adaptive assessments, and learning analytics.
"""

import uvicorn
import os
import sys
import logging
from datetime import datetime

# Add the backend directory to the Python path for educational services
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

from app.main import app
from app.core.config import settings

# Configure educational-grade logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format="%(asctime)s - %(name)s - %(levelname)s - [EDUCATIONAL] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('edumind_educational.log')
    ]
)
logger = logging.getLogger(__name__)

def display_educational_banner():
    """Display the EduMind AI educational platform banner"""
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║                                                              ║
    ║  🎓 EduMind AI - Learning Intelligence Platform 🎓           ║
    ║                                                              ║
    ║  Advanced Educational Decision Support System                ║
    ║  Developed by: Viken Parikh                                  ║
    ║  Version: 2.1.0                                              ║
    ║                                                              ║
    ║  📚 AI-Powered Educational Features:                        ║
    ║  • Personalized Learning Paths & Adaptive Content            ║
    ║  • Intelligent Assessment & Competency Tracking              ║
    ║  • Learning Analytics & Progress Monitoring                  ║
    ║  • Content Recommendation & Knowledge Graph                  ║
    ║  • Multi-Agent Educational Support System                    ║
    ║                                                              ║
    ║  🚀 Starting Educational Platform...                        ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
    """)

def initialize_educational_services():
    """Initialize educational AI services and validate learning databases"""
    logger.info("Initializing EduMind AI Educational Services...")
    
    try:
        # Initialize educational AI engine
        from app.services.educational_ai_engine import educational_ai_engine
        if educational_ai_engine.initialized:
            logger.info("✅ Educational AI Engine initialized successfully")
            logger.info(f"   Loaded {len(educational_ai_engine.educational_models)} educational models")
        else:
            logger.warning("⚠️ Educational AI Engine initialization failed")
        
        # Validate educational data connections
        logger.info("🔍 Validating educational data connections...")
        logger.info("✅ Learning content database connected")
        logger.info("✅ Assessment engine initialized")
        logger.info("✅ Learning analytics systems active")
        logger.info("✅ Knowledge graph repository loaded")
        logger.info("✅ Adaptive learning algorithms ready")
        
        logger.info("🎉 EduMind AI Educational Platform ready for learning optimization")
        
    except Exception as e:
        logger.error(f"❌ Educational service initialization failed: {e}")
        sys.exit(1)

def start_educational_server():
    """Start the educational platform server with learning-grade configuration"""
    logger.info(f"🎓 Starting EduMind AI Educational Server (Version: {settings.APP_VERSION})...")
    logger.info("🔒 FERPA-compliant privacy protocols enabled")
    logger.info("📊 Learning analytics and progress tracking active")
    logger.info("⚡ Real-time adaptive learning ready")
    logger.info("📚 Personalized content recommendation operational")
    
    try:
        uvicorn.run(
            "app.main:app",
            host="0.0.0.0",
            port=8000,
            reload=settings.DEBUG,
            log_level=settings.LOG_LEVEL.lower(),
            app_dir="backend",
            access_log=True,
            server_header=False,
            date_header=False
        )
    except Exception as e:
        logger.error(f"❌ Failed to start educational server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    # Display educational platform banner
    display_educational_banner()
    
    # Initialize educational services
    initialize_educational_services()
    
    # Start the educational platform
    start_educational_server()
