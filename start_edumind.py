#!/usr/bin/env python3
"""
EduMind AI - Learning Intelligence Platform
Startup script for the EduMind AI server
"""

import os
import sys
import subprocess
import webbrowser
import time
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def check_dependencies():
    """Check if required dependencies are installed"""
    try:
        import fastapi
        import uvicorn
        import pandas
        import numpy
        logger.info("✅ All required dependencies are installed")
        return True
    except ImportError as e:
        logger.error(f"❌ Missing dependency: {e}")
        logger.info("Please run: pip install -r requirements.txt")
        return False

def start_server():
    """Start the EduMind AI server"""
    logger.info("🎓 Starting EduMind AI - Learning Intelligence Platform")
    
    # Check if we're in the right directory
    if not Path("backend/app/main.py").exists():
        logger.error("❌ Please run this script from the EduMind-AI directory")
        return False
    
    # Start the server
    try:
        logger.info("🌐 Starting FastAPI server on http://localhost:8000")
        logger.info("📚 API Documentation will be available at http://localhost:8000/docs")
        logger.info("🎓 EduMind AI Learning Intelligence Platform is starting...")
        
        # Start server in background
        server_process = subprocess.Popen([
            sys.executable, "-m", "uvicorn", 
            "backend.app.main:app", 
            "--host", "0.0.0.0", 
            "--port", "8000",
            "--reload"
        ])
        
        # Wait a moment for server to start
        time.sleep(3)
        
        # Open browser
        try:
            webbrowser.open("http://localhost:8000")
            logger.info("🌐 Opened EduMind AI in your default browser")
        except:
            logger.info("🌐 Please open http://localhost:8000 in your browser")
        
        logger.info("🎯 EduMind AI is now running!")
        logger.info("📊 Features available:")
        logger.info("  🎓 Personalized Learning Paths")
        logger.info("  🧠 Adaptive Learning Algorithms") 
        logger.info("  📊 Learning Analytics")
        logger.info("  🎯 Knowledge Gap Analysis")
        logger.info("  📚 Content Recommendation")
        logger.info("  📝 Smart Assessment")
        logger.info("  🔗 Concept Mapping")
        logger.info("  📈 Progress Tracking")
        logger.info("  🎨 Visual Learning Tools")
        logger.info("  ⚡ Real-time Feedback")
        
        logger.info("\n🛑 Press Ctrl+C to stop the server")
        
        # Wait for user to stop
        try:
            server_process.wait()
        except KeyboardInterrupt:
            logger.info("\n🛑 Stopping EduMind AI server...")
            server_process.terminate()
            server_process.wait()
            logger.info("✅ EduMind AI server stopped")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to start server: {e}")
        return False

def main():
    """Main function"""
    print("=" * 80)
    print("🎓 EduMind AI - Learning Intelligence Platform")
    print("=" * 80)
    print("Transforming education through personalized learning intelligence")
    print("that adapts to every student's unique journey.")
    print("=" * 80)
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Start server
    if not start_server():
        sys.exit(1)
    
    print("\n🎉 Thank you for using EduMind AI!")
    print("🎓 Your learning journey is now intelligent and personalized!")

if __name__ == "__main__":
    main()
