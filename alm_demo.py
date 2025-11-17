"""
ALM System Demo
Demonstrates the core functionality of the Audio Language Model without requiring heavy dependencies
"""

class ALMDemo:
    """
    A simplified demonstration of the Audio Language Model capabilities
    """
    
    def __init__(self):
        self.languages = ['english', 'mandarin', 'urdu', 'hindi', 'telugu', 'tamil', 'bangla']
        self.audio_events = ['aircraft_sound', 'announcement', 'footsteps', 'door_slam', 'phone_ring']
        self.speakers = ['speaker_1', 'speaker_2', 'speaker_3']
        self.emotions = ['urgent', 'calm', 'angry', 'happy', 'sad']
    
    def analyze_audio_scene(self, scenario="airport"):
        """
        Simulate audio analysis for a given scenario
        
        Args:
            scenario (str): The audio scenario to analyze
            
        Returns:
            dict: Analysis results
        """
        if scenario == "airport":
            return {
                "speech_recognition": "Flight LH456 to Frankfurt is now boarding at gate B12",
                "audio_events": ["aircraft_sound", "announcement", "footsteps"],
                "speakers": ["speaker_1", "speaker_2"],
                "paralinguistics": "urgent",
                "scene_interpretation": "The aircraft sound and announcement suggest that the person is in an airport boarding area. The urgency in the voice indicates they may be concerned about their flight."
            }
        elif scenario == "office":
            return {
                "speech_recognition": "Please send me the quarterly report by end of day",
                "audio_events": ["keyboard_typing", "phone_ring", "door_slam"],
                "speakers": ["speaker_1"],
                "paralinguistics": "calm",
                "scene_interpretation": "This appears to be an office environment with typical office sounds. The person seems to be giving instructions in a calm manner."
            }
        else:
            return {
                "speech_recognition": "Hello, how are you today?",
                "audio_events": ["footsteps"],
                "speakers": ["speaker_1"],
                "paralinguistics": "calm",
                "scene_interpretation": "A general conversation scenario with minimal background noise."
            }
    
    def ask_question(self, question, scenario="airport"):
        """
        Simulate question answering about an audio scene
        
        Args:
            question (str): Question about the audio
            scenario (str): The audio scenario
            
        Returns:
            str: Answer to the question
        """
        analysis = self.analyze_audio_scene(scenario)
        
        if "what" in question.lower() and "happen" in question.lower():
            return analysis["scene_interpretation"]
        elif "who" in question.lower() or "speaker" in question.lower():
            return f"Identified {len(analysis['speakers'])} speaker(s): {', '.join(analysis['speakers'])}"
        elif "language" in question.lower():
            return "The primary language detected is English"
        elif "emotion" in question.lower() or "feel" in question.lower():
            return f"The speaker's tone is {analysis['paralinguistics']}"
        else:
            return "Based on the audio analysis, the scene appears to be a typical airport boarding scenario with announcements and aircraft sounds."

def main():
    """
    Main demo function
    """
    print("Audio Language Model (ALM) Demo")
    print("=" * 40)
    print("This demo simulates the capabilities of the ALM system")
    print()
    
    # Create ALM demo instance
    alm = ALMDemo()
    
    # Simulate analysis of an airport scenario
    print("Analyzing airport audio scenario...")
    print("-" * 30)
    results = alm.analyze_audio_scene("airport")
    
    for key, value in results.items():
        print(f"{key.replace('_', ' ').title()}: {value}")
    
    print("\n" + "=" * 40)
    print("Question Answering Demo")
    print("-" * 30)
    
    # Simulate question answering
    questions = [
        "What is happening in this audio?",
        "Who are the speakers?",
        "What language is being spoken?",
        "What is the speaker's emotion?"
    ]
    
    for question in questions:
        answer = alm.ask_question(question, "airport")
        print(f"Q: {question}")
        print(f"A: {answer}")
        print()

if __name__ == "__main__":
    main()