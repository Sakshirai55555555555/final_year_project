def recommend(gait_class):
    """Generate medical recommendations based on the detected abnormality."""
    recommendations = {
        "Normal": {
            "abnormality": "Normal Gait",
            "explanation": "Your gait appears normal. No corrective action needed.",
            "exercises": [],
            "advice": "Keep maintaining a healthy posture and stay active!"
        },
        "Limping": {
            "abnormality": "Limping",
            "explanation": "Limping may occur due to muscle weakness or injury.",
            "exercises": [
                {"name": "Leg Raises", "description": "Strengthen quadriceps with leg raises."},
                {"name": "Calf Stretches", "description": "Improve flexibility with calf stretches."}
            ],
            "advice": "If pain persists, consult a physiotherapist."
        },
        "No Arm Swing": {
            "abnormality": "No Arm Swing",
            "explanation": "Lack of arm swing might be due to muscle stiffness or imbalance.",
            "exercises": [{"name": "Arm Circles", "description": "Improve shoulder mobility with arm circles."}],
            "advice": "Focus on a natural arm swing while walking."
        }
    }

    return recommendations.get(gait_class, {
        "abnormality": "Unknown",
        "explanation": "No specific information available.",
        "exercises": [],
        "advice": "Consult a healthcare professional for accurate diagnosis."
    })
