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
        }
    }

    return recommendations.get(gait_class, {
        "abnormality": "Unknown",
        "explanation": "No specific information available.",
        "exercises": [],
        "advice": "Consult a healthcare professional for accurate diagnosis."
    })
