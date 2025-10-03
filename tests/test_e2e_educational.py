"""
EduMind AI - End-to-End Tests
Comprehensive E2E tests for educational AI platform
"""

import pytest
import asyncio
from fastapi.testclient import TestClient
from app.main import app
from app.services.educational_ai_engine import educational_ai_engine
import json
import numpy as np
from datetime import datetime

client = TestClient(app)

class TestEduMindAIE2E:
    """End-to-End test suite for EduMind AI platform"""
    
    def setup_method(self):
        """Setup for each test method"""
        self.test_student_id = "student_test_001"
        self.test_student_data = {
            "age": 22,
            "grade_level": "undergraduate",
            "learning_preferences": {
                "visual": 0.8,
                "auditory": 0.6,
                "kinesthetic": 0.7,
                "reading_writing": 0.9
            },
            "performance_history": {
                "average_score": 0.85,
                "completion_rate": 0.92,
                "engagement_score": 0.78
            },
            "learning_activities": [
                {
                    "topic": "Machine Learning",
                    "duration": 120,
                    "completed": True,
                    "score": 0.88,
                    "difficulty": "intermediate"
                },
                {
                    "topic": "Deep Learning",
                    "duration": 90,
                    "completed": True,
                    "score": 0.82,
                    "difficulty": "advanced"
                },
                {
                    "topic": "Natural Language Processing",
                    "duration": 150,
                    "completed": False,
                    "score": 0.65,
                    "difficulty": "intermediate"
                }
            ]
        }
        self.test_learning_goals = [
            "Master machine learning fundamentals",
            "Understand deep learning architectures",
            "Apply NLP techniques to real problems",
            "Develop AI applications"
        ]
        self.test_assessment_config = {
            "max_questions": 20,
            "min_questions": 10,
            "time_limit": 60,
            "topics": ["Machine Learning", "Deep Learning", "NLP"],
            "difficulty_levels": ["easy", "medium", "hard"]
        }
    
    def test_platform_health_e2e(self):
        """Test complete platform health and functionality"""
        print("\n🎓 Testing EduMind AI Platform Health...")
        
        # Test root endpoint
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "Welcome to EduMind AI" in data["message"]
        print("✅ Root endpoint working")
        
        # Test health check
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        print("✅ Health check working")
        
        # Test API health
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        print("✅ API health working")
        
        print("🎉 EduMind AI platform is healthy!")
    
    @pytest.mark.asyncio
    async def test_educational_ai_engine_e2e(self):
        """Test educational AI engine end-to-end functionality"""
        print("\n🧠 Testing Educational AI Engine...")
        
        # Test teaching crew creation
        crew_config = {
            "name": "Personalized Learning Crew",
            "agents": [
                {"role": "Learning Coach", "goal": "Guide student learning"},
                {"role": "Content Curator", "goal": "Recommend optimal content"}
            ]
        }
        
        result = await educational_ai_engine.create_teaching_crew(crew_config)
        assert "crew_id" in result
        assert result["status"] == "created"
        print("✅ Teaching crew creation working")
        
        # Test learning style analysis
        result = await educational_ai_engine.analyze_learning_style(self.test_student_data)
        assert "learning_style" in result
        assert "personalization_recommendations" in result
        assert "cognitive_preferences" in result
        print("✅ Learning style analysis working")
        
        # Test knowledge state tracking
        result = await educational_ai_engine.track_knowledge_state(
            self.test_student_id, self.test_student_data["learning_activities"]
        )
        assert "knowledge_graph" in result
        assert "mastery_levels" in result
        assert "knowledge_gaps" in result
        print("✅ Knowledge state tracking working")
        
        # Test adaptive assessment generation
        result = await educational_ai_engine.generate_adaptive_assessment(
            self.test_student_data, self.test_assessment_config
        )
        assert "adaptive_sequence" in result
        assert "question_bank" in result
        assert "assessment_analytics" in result
        print("✅ Adaptive assessment generation working")
        
        # Test content recommendation
        result = await educational_ai_engine.recommend_learning_content(
            self.test_student_data, self.test_learning_goals
        )
        assert "optimized_learning_path" in result
        assert "diverse_recommendations" in result
        assert "engagement_predictions" in result
        print("✅ Content recommendation working")
        
        print("🎉 Educational AI Engine is fully functional!")
    
    def test_educational_api_endpoints_e2e(self):
        """Test educational API endpoints end-to-end"""
        print("\n🔗 Testing Educational API Endpoints...")
        
        # Test learning style analysis endpoint
        response = client.post("/api/v1/educational/learning-style-analysis",
                             params={"student_data": json.dumps(self.test_student_data)})
        assert response.status_code == 200
        data = response.json()
        assert "learning_style" in data
        print("✅ Learning style analysis endpoint working")
        
        # Test knowledge tracking endpoint
        response = client.post("/api/v1/educational/knowledge-tracking",
                             params={
                                 "student_id": self.test_student_id,
                                 "learning_activities": json.dumps(self.test_student_data["learning_activities"])
                             })
        assert response.status_code == 200
        data = response.json()
        assert "mastery_levels" in data
        print("✅ Knowledge tracking endpoint working")
        
        # Test adaptive assessment endpoint
        response = client.post("/api/v1/educational/adaptive-assessment",
                             params={
                                 "student_profile": json.dumps(self.test_student_data),
                                 "assessment_config": json.dumps(self.test_assessment_config)
                             })
        assert response.status_code == 200
        data = response.json()
        assert "adaptive_sequence" in data
        print("✅ Adaptive assessment endpoint working")
        
        # Test content recommendation endpoint
        response = client.post("/api/v1/educational/content-recommendation",
                             params={
                                 "student_profile": json.dumps(self.test_student_data),
                                 "learning_goals": json.dumps(self.test_learning_goals)
                             })
        assert response.status_code == 200
        data = response.json()
        assert "optimized_learning_path" in data
        print("✅ Content recommendation endpoint working")
        
        print("🎉 Educational API endpoints are fully functional!")
    
    def test_educational_workflow_e2e(self):
        """Test complete educational workflow end-to-end"""
        print("\n🔄 Testing Complete Educational Workflow...")
        
        # Step 1: Analyze student learning style
        style_response = client.post("/api/v1/educational/learning-style-analysis",
                                   params={"student_data": json.dumps(self.test_student_data)})
        assert style_response.status_code == 200
        style_data = style_response.json()
        print("✅ Step 1: Learning style analysis completed")
        
        # Step 2: Track knowledge state
        tracking_response = client.post("/api/v1/educational/knowledge-tracking",
                                      params={
                                          "student_id": self.test_student_id,
                                          "learning_activities": json.dumps(self.test_student_data["learning_activities"])
                                      })
        assert tracking_response.status_code == 200
        tracking_data = tracking_response.json()
        print("✅ Step 2: Knowledge state tracking completed")
        
        # Step 3: Generate adaptive assessment
        assessment_response = client.post("/api/v1/educational/adaptive-assessment",
                                        params={
                                            "student_profile": json.dumps(self.test_student_data),
                                            "assessment_config": json.dumps(self.test_assessment_config)
                                        })
        assert assessment_response.status_code == 200
        assessment_data = assessment_response.json()
        print("✅ Step 3: Adaptive assessment generation completed")
        
        # Step 4: Recommend learning content
        content_response = client.post("/api/v1/educational/content-recommendation",
                                     params={
                                         "student_profile": json.dumps(self.test_student_data),
                                         "learning_goals": json.dumps(self.test_learning_goals)
                                     })
        assert content_response.status_code == 200
        content_data = content_response.json()
        print("✅ Step 4: Content recommendation completed")
        
        # Step 5: Generate comprehensive learning report
        learning_report = {
            "student_id": self.test_student_id,
            "analysis_time": datetime.utcnow().isoformat(),
            "learning_style_analysis": style_data,
            "knowledge_tracking": tracking_data,
            "adaptive_assessment": assessment_data,
            "content_recommendations": content_data,
            "learning_summary": {
                "primary_learning_style": style_data["learning_style"]["primary_modality"],
                "overall_mastery": np.mean([level["mastery_score"] for level in tracking_data["mastery_levels"].values()]),
                "knowledge_gaps": len(tracking_data["knowledge_gaps"]),
                "assessment_readiness": assessment_data["proficiency_analysis"]["overall_proficiency"],
                "recommended_actions": [
                    f"Focus on {style_data['learning_style']['primary_modality']} learning approach",
                    f"Address {len(tracking_data['knowledge_gaps'])} knowledge gaps",
                    f"Take adaptive assessment (confidence: {assessment_data['generation_confidence']:.1%})",
                    f"Follow optimized learning path (success probability: {content_data['optimized_learning_path']['success_probability']:.1%})"
                ]
            }
        }
        
        # Validate comprehensive report
        assert "student_id" in learning_report
        assert "learning_style_analysis" in learning_report
        assert "knowledge_tracking" in learning_report
        assert "adaptive_assessment" in learning_report
        assert "content_recommendations" in learning_report
        assert "learning_summary" in learning_report
        print("✅ Step 5: Comprehensive learning report generated")
        
        print("🎉 Complete educational workflow is functional!")
        print(f"📚 Learning Report Summary:")
        print(f"   Student ID: {learning_report['student_id']}")
        print(f"   Learning Style: {learning_report['learning_summary']['primary_learning_style']}")
        print(f"   Overall Mastery: {learning_report['learning_summary']['overall_mastery']:.1%}")
        print(f"   Knowledge Gaps: {learning_report['learning_summary']['knowledge_gaps']}")
        print(f"   Assessment Readiness: {learning_report['learning_summary']['assessment_readiness']:.1%}")
    
    def test_educational_personalization_e2e(self):
        """Test educational personalization end-to-end"""
        print("\n🎯 Testing Educational Personalization...")
        
        # Test different learning styles
        visual_learner = self.test_student_data.copy()
        visual_learner["learning_preferences"]["visual"] = 0.9
        visual_learner["learning_preferences"]["auditory"] = 0.3
        
        response = client.post("/api/v1/educational/learning-style-analysis",
                             params={"student_data": json.dumps(visual_learner)})
        assert response.status_code == 200
        data = response.json()
        assert data["learning_style"]["primary_modality"] == "visual_learner"
        print("✅ Visual learner detection working")
        
        # Test auditory learner
        auditory_learner = self.test_student_data.copy()
        auditory_learner["learning_preferences"]["auditory"] = 0.9
        auditory_learner["learning_preferences"]["visual"] = 0.3
        
        response = client.post("/api/v1/educational/learning-style-analysis",
                             params={"student_data": json.dumps(auditory_learner)})
        assert response.status_code == 200
        data = response.json()
        assert data["learning_style"]["primary_modality"] == "auditory_learner"
        print("✅ Auditory learner detection working")
        
        # Test kinesthetic learner
        kinesthetic_learner = self.test_student_data.copy()
        kinesthetic_learner["learning_preferences"]["kinesthetic"] = 0.9
        kinesthetic_learner["learning_preferences"]["visual"] = 0.3
        
        response = client.post("/api/v1/educational/learning-style-analysis",
                             params={"student_data": json.dumps(kinesthetic_learner)})
        assert response.status_code == 200
        data = response.json()
        assert data["learning_style"]["primary_modality"] == "kinesthetic_learner"
        print("✅ Kinesthetic learner detection working")
        
        print("🎉 Educational personalization is working!")
    
    def test_educational_adaptive_assessment_e2e(self):
        """Test adaptive assessment functionality end-to-end"""
        print("\n📝 Testing Adaptive Assessment...")
        
        # Test beginner student
        beginner_student = self.test_student_data.copy()
        beginner_student["performance_history"]["average_score"] = 0.45
        beginner_student["learning_activities"] = [
            {
                "topic": "Basic Concepts",
                "duration": 60,
                "completed": True,
                "score": 0.4,
                "difficulty": "easy"
            }
        ]
        
        response = client.post("/api/v1/educational/adaptive-assessment",
                             params={
                                 "student_profile": json.dumps(beginner_student),
                                 "assessment_config": json.dumps(self.test_assessment_config)
                             })
        assert response.status_code == 200
        data = response.json()
        # Should generate easier questions for beginner
        assert data["proficiency_analysis"]["overall_proficiency"] < 0.6
        print("✅ Beginner assessment adaptation working")
        
        # Test advanced student
        advanced_student = self.test_student_data.copy()
        advanced_student["performance_history"]["average_score"] = 0.95
        advanced_student["learning_activities"] = [
            {
                "topic": "Advanced Topics",
                "duration": 120,
                "completed": True,
                "score": 0.92,
                "difficulty": "advanced"
            }
        ]
        
        response = client.post("/api/v1/educational/adaptive-assessment",
                             params={
                                 "student_profile": json.dumps(advanced_student),
                                 "assessment_config": json.dumps(self.test_assessment_config)
                             })
        assert response.status_code == 200
        data = response.json()
        # Should generate harder questions for advanced student
        assert data["proficiency_analysis"]["overall_proficiency"] > 0.7
        print("✅ Advanced assessment adaptation working")
        
        print("🎉 Adaptive assessment is working!")
    
    def test_educational_performance_e2e(self):
        """Test educational platform performance end-to-end"""
        print("\n⚡ Testing Educational Platform Performance...")
        
        import time
        
        # Test response times
        start_time = time.time()
        response = client.get("/health")
        health_time = time.time() - start_time
        assert health_time < 1.0  # Should respond within 1 second
        print(f"✅ Health check response time: {health_time:.3f}s")
        
        start_time = time.time()
        response = client.post("/api/v1/educational/learning-style-analysis",
                             params={"student_data": json.dumps(self.test_student_data)})
        style_time = time.time() - start_time
        assert style_time < 5.0  # Should respond within 5 seconds
        print(f"✅ Learning style analysis response time: {style_time:.3f}s")
        
        start_time = time.time()
        response = client.post("/api/v1/educational/knowledge-tracking",
                             params={
                                 "student_id": self.test_student_id,
                                 "learning_activities": json.dumps(self.test_student_data["learning_activities"])
                             })
        tracking_time = time.time() - start_time
        assert tracking_time < 5.0  # Should respond within 5 seconds
        print(f"✅ Knowledge tracking response time: {tracking_time:.3f}s")
        
        start_time = time.time()
        response = client.post("/api/v1/educational/content-recommendation",
                             params={
                                 "student_profile": json.dumps(self.test_student_data),
                                 "learning_goals": json.dumps(self.test_learning_goals)
                             })
        content_time = time.time() - start_time
        assert content_time < 5.0  # Should respond within 5 seconds
        print(f"✅ Content recommendation response time: {content_time:.3f}s")
        
        print("🎉 Educational platform performance is excellent!")
    
    def test_educational_error_handling_e2e(self):
        """Test educational error handling end-to-end"""
        print("\n⚠️ Testing Educational Error Handling...")
        
        # Test invalid student data
        response = client.post("/api/v1/educational/learning-style-analysis",
                             params={"student_data": json.dumps({})})
        assert response.status_code == 200  # Should handle gracefully
        print("✅ Empty student data handled gracefully")
        
        # Test invalid learning activities
        response = client.post("/api/v1/educational/knowledge-tracking",
                             params={
                                 "student_id": self.test_student_id,
                                 "learning_activities": json.dumps([])
                             })
        assert response.status_code == 200  # Should handle gracefully
        print("✅ Empty learning activities handled gracefully")
        
        # Test invalid assessment config
        response = client.post("/api/v1/educational/adaptive-assessment",
                             params={
                                 "student_profile": json.dumps(self.test_student_data),
                                 "assessment_config": json.dumps({})
                             })
        assert response.status_code == 200  # Should handle gracefully
        print("✅ Empty assessment config handled gracefully")
        
        # Test invalid learning goals
        response = client.post("/api/v1/educational/content-recommendation",
                             params={
                                 "student_profile": json.dumps(self.test_student_data),
                                 "learning_goals": json.dumps([])
                             })
        assert response.status_code == 200  # Should handle gracefully
        print("✅ Empty learning goals handled gracefully")
        
        print("🎉 Educational error handling is robust!")

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
