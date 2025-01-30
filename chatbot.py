import openai
from typing import Dict, List, Optional
import logging
import json
from fuzzywuzzy import fuzz
import os

class RPIChatbot:
    def __init__(self):
        self.context = {
            'current_topic': None,
            'current_subtopic': None,
            'conversation_history': []
        }
        
        # Load knowledge base
        with open('data/knowledge_base.json', 'r') as f:
            self.knowledge = json.load(f)
            
        # Configure OpenAI
        openai.api_key = os.getenv('OPENAI_API_KEY')
        
    def process_input(self, user_input: str) -> str:
        """Process user input and generate appropriate response"""
        # Add input to history
        self.context['conversation_history'].append({"role": "user", "content": user_input})
        
        # Check for follow-up questions about the current topic
        if self.context['current_topic']:
            response = self._handle_followup(user_input)
            if response:
                self.context['conversation_history'].append({"role": "assistant", "content": response})
                return response
        
        # Try fuzzy matching first
        landmark = self._fuzzy_match_landmark(user_input)
        
        if not landmark:
            # If fuzzy matching fails, try GPT for noisy input handling
            try:
                gpt_analysis = self._analyze_noisy_input(user_input)
                if gpt_analysis.get('landmark'):
                    return self._handle_noisy_input(gpt_analysis['original'], gpt_analysis['landmark'])
            except Exception as e:
                logging.error(f"Error in GPT analysis: {e}")
            
            return "I'm not sure which RPI landmark you're asking about. Could you specify one of: Russell Sage Laboratory, West Hall, RPI Union, Folsom Library, or EMPAC?"
        else:
            self.context['current_topic'] = landmark
            info = self.knowledge['landmarks'].get(landmark, {})
            return self._generate_basic_response(landmark, info)

    def _handle_clear_reference(self, landmark: str) -> str:
        """Handle clear reference to a landmark"""
        self.context['current_topic'] = landmark
        info = self.knowledge['landmarks'].get(landmark, {})
        return self._generate_basic_response(landmark, info)

    def _handle_ambiguous_reference(self, possible_landmarks: List[str]) -> str:
        """Handle ambiguous reference to multiple landmarks"""
        response = "I'm not sure which landmark you mean. Are you asking about:\n"
        for landmark in possible_landmarks:
            response += f"- {self.knowledge['landmarks'][landmark]['name']}\n"
        response += "\nCould you please clarify?"
        return response

    def _handle_noisy_input(self, original_input: str, detected_landmark: str) -> str:
        """Handle noisy/misspelled input"""
        name = self.knowledge['landmarks'][detected_landmark]['name']
        response = f"I understand you're asking about {name}. "
        self.context['current_topic'] = detected_landmark
        info = self.knowledge['landmarks'][detected_landmark]
        response += self._generate_basic_response(detected_landmark, info)
        return response

    def _generate_basic_response(self, landmark: str, info: Dict) -> str:
        """Generate a basic response about a landmark"""
        if not info:
            return "I have some information about that landmark, but I'm having trouble accessing it right now."
        
        name = info.get('name', 'this landmark')
        response_parts = []
        
        # Add basic information
        if 'built' in info:
            response_parts.append(f"{name} was built in {info['built']}.")
        elif 'established' in info:
            response_parts.append(f"{name} was established in {info['established']}.")
        elif 'dedicated' in info:
            response_parts.append(f"{name} was dedicated in {info['dedicated']}.")
        
        # Add significance if available
        if 'significance' in info:
            response_parts.append(f"It is notable for being {info['significance']}.")
        
        # Add context-aware follow-up prompt
        if landmark == "rpi_union":
            response_parts.append("Would you like to know more about its student activities, events, or facilities?")
        else:
            response_parts.append("Would you like to know more about its history, architecture, or current use?")
        
        return " ".join(response_parts)

    def _fuzzy_match_landmark(self, query: str) -> Optional[str]:
        """Match potentially noisy landmark references to known landmarks"""
        landmarks = {
            "russell sage": "russell_sage",
            "sage lab": "russell_sage",
            "sage laboratory": "russell_sage",
            "russel": "russell_sage",
            "russell": "russell_sage",
            "sage": "russell_sage",
            "rsl": "russell_sage",
            "west hall": "west_hall",
            "west": "west_hall",
            "wh": "west_hall",
            "rpi union": "rpi_union",
            "student union": "rpi_union",
            "union": "rpi_union",
            "campus union": "rpi_union",
            "folsom": "folsom_library",
            "library": "folsom_library",
            "folsom lib": "folsom_library",
            "folsum": "folsom_library",
            "folsam": "folsom_library",
            "lib": "folsom_library",
            "empac": "empac",
            "experimental media": "empac",
            "performing arts": "empac",
            "arts center": "empac",
            "experimental": "empac",
            "media center": "empac"
        }
        
        # Normalize query
        query = query.lower().strip()
        
        # Remove common filler words
        query = query.replace("tell me about", "").replace("what about", "").replace("where is", "").strip()
        
        # Try direct matches first
        for key, value in landmarks.items():
            if query in key or key in query:
                return value
                
        # Try fuzzy matching if no direct match
        best_match = None
        best_ratio = 0
        
        for key in landmarks.keys():
            ratio = fuzz.partial_ratio(query, key)
            if ratio > best_ratio and ratio > 70:  # Lower threshold for more lenient matching
                best_ratio = ratio
                best_match = landmarks[key]
                
        return best_match

    def _handle_followup(self, user_input: str) -> Optional[str]:
        """Handle follow-up questions about the current landmark"""
        query = user_input.lower()
        info = self.knowledge['landmarks'].get(self.context['current_topic'], {})
        
        # Special handling for RPI Union
        if self.context['current_topic'] == "rpi_union":
            if any(word in query for word in ['event', 'activities', 'programs']):
                return self._get_events_info(info)
            elif any(word in query for word in ['facilities', 'rooms', 'spaces']):
                return self._get_facilities_info(info)
            elif any(word in query for word in ['history', 'background', 'past']):
                return self._get_history_info(info)
            return None
        
        # Standard handling for other landmarks
        if any(word in query for word in ['history', 'background', 'past', 'origin']):
            return self._get_history_info(info)
        elif any(word in query for word in ['architecture', 'design', 'building', 'structure']):
            return self._get_architecture_info(info)
        elif any(word in query for word in ['current', 'now', 'today', 'use', 'purpose']):
            return self._get_current_use_info(info)
        
        return None
        
    def _get_history_info(self, info: Dict) -> str:
        """Generate response about landmark's history"""
        response_parts = []
        name = info.get('name', 'This landmark')
        
        if 'history' in info:
            history = info['history']
            # Handle different history formats
            if 'evolution' in history:
                response_parts.append(f"{history['evolution']}")
            if 'origins' in history:
                response_parts.append(f"Its origins date back to {history['origins']}.")
            if 'original_purpose' in history:
                response_parts.append(f"{name}'s original purpose was as {history['original_purpose']}.")
            if 'builder' in history:
                response_parts.append(f"It was built by {history['builder']}.")
            if 'timeline' in history:
                events = history['timeline']
                response_parts.append("Key events in its history include:")
                for event in events:
                    response_parts.append(f"- {event['year']}: {event['event']}")
            if 'significance' in history:
                response_parts.append(f"It is historically significant as {history['significance']}.")
            if 'namesake' in info:  # Some landmarks have namesake info at top level
                namesake = info['namesake']
                if isinstance(namesake, dict):
                    response_parts.append(f"It was named after {namesake['name']}")
                    if 'role' in namesake:
                        response_parts.append(f"who was {namesake['role']}")
                    if 'years' in namesake:
                        response_parts.append(f"from {namesake['years']}")
        
        if not response_parts:
            return f"I don't have detailed historical information about {name}, but you can ask about its architecture or current use."
        
        return " ".join(response_parts).replace("..", ".") + "."
        
    def _get_architecture_info(self, info: Dict) -> str:
        """Generate response about landmark's architecture"""
        response_parts = []
        name = info.get('name', 'This landmark')
        
        if 'architecture' in info:
            arch = info['architecture']
            if 'style' in arch:
                response_parts.append(f"{name} features {arch['style']} architecture.")
            if 'features' in arch:
                response_parts.append(f"Notable architectural features include: {', '.join(arch['features'])}.")
            if 'architect' in arch:
                response_parts.append(f"It was designed by {arch['architect']}.")
        
        if not response_parts:
            return f"I don't have detailed architectural information about {name}, but you can ask about its history or current use."
        
        return " ".join(response_parts)
        
    def _get_current_use_info(self, info: Dict) -> str:
        """Generate response about landmark's current use"""
        response_parts = []
        name = info.get('name', 'This landmark')
        
        # Handle standard current_use structure
        if 'current_use' in info:
            current = info['current_use']
            if isinstance(current, dict):
                if 'departments' in current:
                    response_parts.append(f"{name} currently houses {', '.join(current['departments'])}.")
                elif 'department' in current:
                    response_parts.append(f"{name} currently houses the {current['department']}.")
                if 'facilities' in current:
                    response_parts.append(f"Its facilities include: {', '.join(current['facilities'])}.")
        
        # Handle RPI Union specific structure
        if 'features' in info:
            features = info['features']
            if isinstance(features, dict):
                if 'student_activities' in features:
                    activities = features['student_activities']
                    if 'clubs' in activities:
                        response_parts.append(f"It hosts {activities['clubs']}.")
                    if 'types' in activities:
                        response_parts.append(f"These include {', '.join(activities['types'])}.")
                if 'facilities' in features:
                    response_parts.append(f"Facilities include: {', '.join(features['facilities'])}.")

        if 'events' in info:
            response_parts.append(f"Regular events include: {', '.join(info['events'])}.")

        if 'management' in info:
            response_parts.append(f"It is a {info['management']}.")
        
        if not response_parts:
            return f"I don't have detailed information about {name}'s current use, but you can ask about its history or architecture."
        
        return " ".join(response_parts)

    def _get_events_info(self, info: Dict) -> str:
        """Generate response about Union's events"""
        response_parts = []
        name = info.get('name', 'The Union')
        
        if 'events' in info:
            response_parts.append(f"{name} hosts various events including: {', '.join(info['events'])}.")
        
        if 'features' in info and 'student_activities' in info['features']:
            activities = info['features']['student_activities']
            if 'clubs' in activities:
                response_parts.append(f"It supports {activities['clubs']}.")
            if 'types' in activities:
                response_parts.append(f"These include {', '.join(activities['types'])}.")
        
        return " ".join(response_parts)

    def _get_facilities_info(self, info: Dict) -> str:
        """Generate response about Union's facilities"""
        response_parts = []
        name = info.get('name', 'The Union')
        
        if 'features' in info and 'facilities' in info['features']:
            response_parts.append(f"{name}'s facilities include: {', '.join(info['features']['facilities'])}.")
        
        return " ".join(response_parts)

    def _analyze_noisy_input(self, user_input: str) -> Dict:
        """Use GPT to analyze potentially noisy or ambiguous input"""
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": """
                        You are analyzing user input for an RPI landmarks chatbot.
                        The landmarks are: Russell Sage Laboratory, West Hall, RPI Union, Folsom Library, and EMPAC.
                        If the input seems to be referring to one of these landmarks but is misspelled or unclear,
                        identify which landmark they likely mean.
                        
                        Return a JSON with:
                        {
                            "landmark": "identified landmark key or null",
                            "original": "what they typed",
                            "confidence": "high/medium/low",
                            "reasoning": "brief explanation"
                        }
                    """},
                    {"role": "user", "content": user_input}
                ],
                temperature=0.3
            )
            
            return json.loads(response.choices[0].message['content'])
        except Exception as e:
            logging.error(f"Error in GPT analysis: {e}")
            return {} 