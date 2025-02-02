import openai
from typing import Dict, List, Optional
import logging
import json
from fuzzywuzzy import fuzz
import os

class RPIChatbot:
    """
    A chatbot specialized in handling queries about RPI (Rensselaer Polytechnic Institute) landmarks.
    
    This class implements a context-aware conversation system that can:
    - Process natural language queries about RPI landmarks
    - Handle fuzzy matching for misspelled or unclear references
    - Maintain conversation context for follow-up questions
    - Generate detailed responses about landmark history, architecture, and current use
    - Integrate with OpenAI's GPT for handling ambiguous queries
    
    The chatbot uses a JSON-based knowledge base and implements various matching and
    response generation strategies to provide accurate and contextual information.
    """

    def __init__(self):
        """
        Initialize the chatbot with an empty context and load the knowledge base.
        Sets up OpenAI API integration and initializes conversation tracking.
        """
        # Track current conversation state and history
        self.context = {
            'current_topic': None,  # Currently discussed landmark
            'current_subtopic': None,  # Current aspect (history, architecture, etc.)
            'conversation_history': []  # Full conversation log
        }
        
        # Load landmark information from JSON file
        # Expected format: {'landmarks': {'landmark_key': {'name': str, 'built': str, ...}}}
        with open('data/knowledge_base.json', 'r') as f:
            self.knowledge = json.load(f)
            
        # Configure OpenAI API for advanced query processing
        openai.api_key = os.getenv('OPENAI_API_KEY')
        
    def process_input(self, user_input: str) -> str:
        """
        Process user input and generate an appropriate response.
        
        This is the main entry point for handling user queries. It implements a multi-stage
        processing pipeline:
        1. Context check for follow-up questions
        2. Fuzzy matching for landmark identification
        3. GPT-based analysis for ambiguous inputs
        4. Response generation based on matched landmark
        
        Args:
            user_input: The raw text input from the user
            
        Returns:
            str: Generated response based on the input and current context
        """
        # Log user input in conversation history
        self.context['conversation_history'].append({"role": "user", "content": user_input})
        
        # Check if this is a follow-up question about the current topic
        if self.context['current_topic']:
            response = self._handle_followup(user_input)
            if response:
                self.context['conversation_history'].append({"role": "assistant", "content": response})
                return response
        
        # Attempt fuzzy matching to identify landmark references
        match_result = self._fuzzy_match_landmark(user_input)
        
        if not match_result:
            # If fuzzy matching fails, attempt GPT-based analysis
            try:
                gpt_analysis = self._analyze_noisy_input(user_input)
                if gpt_analysis.get('landmark'):
                    return self._handle_noisy_input(gpt_analysis['original'], gpt_analysis['landmark'])
            except Exception as e:
                logging.error(f"Error in GPT analysis: {e}")
            
            # If all matching attempts fail, provide guidance
            return "I'm not sure which RPI landmark you're asking about. Could you specify one of: Russell Sage Laboratory, West Hall, RPI Union, Folsom Library, or EMPAC?"
        else:
            # Process successful match
            landmark = match_result['landmark']
            self.context['current_topic'] = landmark
            info = self.knowledge['landmarks'].get(landmark, {})
            
            # Handle fuzzy vs exact matches differently
            if match_result['is_fuzzy']:
                return f"I think you might be referring to {info['name']}. " + self._generate_basic_response(landmark, info)
            else:
                return self._generate_basic_response(landmark, info)

    def _handle_clear_reference(self, landmark: str) -> str:
        """
        Handle unambiguous references to landmarks.
        
        Args:
            landmark: The identified landmark key
            
        Returns:
            str: Generated response about the landmark
        """
        self.context['current_topic'] = landmark
        info = self.knowledge['landmarks'].get(landmark, {})
        return self._generate_basic_response(landmark, info)

    def _handle_ambiguous_reference(self, possible_landmarks: List[str]) -> str:
        """
        Handle cases where multiple landmarks might match the query.
        Generates a clarification request listing possible matches.
        
        Args:
            possible_landmarks: List of potential landmark matches
            
        Returns:
            str: Response asking for clarification
        """
        response = "I'm not sure which landmark you mean. Are you asking about:\n"
        for landmark in possible_landmarks:
            response += f"- {self.knowledge['landmarks'][landmark]['name']}\n"
        response += "\nCould you please clarify?"
        return response

    def _handle_noisy_input(self, original_input: str, detected_landmark: str) -> str:
        """
        Handle inputs with potential misspellings or unclear references.
        Confirms the interpreted landmark and provides information.
        
        Args:
            original_input: The user's original query
            detected_landmark: The landmark identified through analysis
            
        Returns:
            str: Confirmation and information about the detected landmark
        """
        name = self.knowledge['landmarks'][detected_landmark]['name']
        response = f"I think you might be referring to {name}. "
        self.context['current_topic'] = detected_landmark
        info = self.knowledge['landmarks'][detected_landmark]
        response += self._generate_basic_response(detected_landmark, info)
        return response

    def _generate_basic_response(self, landmark: str, info: Dict) -> str:
        """
        Generate a standard response about a landmark including basic information
        and a context-appropriate follow-up prompt.
        
        Args:
            landmark: The landmark identifier
            info: Dictionary containing landmark information
            
        Returns:
            str: Formatted response with basic information and follow-up prompt
        """
        if not info:
            return "I have some information about that landmark, but I'm having trouble accessing it right now."
        
        name = info.get('name', 'this landmark')
        response_parts = []
        
        # Add establishment information using available date fields
        if 'built' in info:
            response_parts.append(f"{name} was built in {info['built']}.")
        elif 'established' in info:
            response_parts.append(f"{name} was established in {info['established']}.")
        elif 'dedicated' in info:
            response_parts.append(f"{name} was dedicated in {info['dedicated']}.")
        
        # Add significance if available
        if 'significance' in info:
            response_parts.append(f"It is notable for being {info['significance']}.")
        
        # Add context-specific follow-up prompt
        if landmark == "rpi_union":
            response_parts.append("Would you like to know more about its student activities, events, or facilities?")
        else:
            response_parts.append("Would you like to know more about its history, architecture, or current use?")
        
        return " ".join(response_parts)

    def _fuzzy_match_landmark(self, query: str) -> Optional[Dict]:
        """
        Match user input to known landmarks using fuzzy string matching.
        Handles variations in spelling, partial matches, and common abbreviations.
        
        Implementation uses a comprehensive alias dictionary and the fuzzywuzzy
        library for string similarity matching.
        
        Args:
            query: The user's input text
            
        Returns:
            Optional[Dict]: Dictionary with matched landmark and match type,
                          or None if no match found
        """
        # Comprehensive alias dictionary mapping various references to landmark keys
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
        
        # Normalize query for matching
        query = query.lower().strip()
        original_query = query
        
        # Remove common filler phrases
        query = query.replace("tell me about", "").replace("what about", "").replace("where is", "").strip()
        
        # Try exact matches first
        for key, value in landmarks.items():
            if query in key or key in query:
                return {"landmark": value, "is_fuzzy": False}
                
        # Fall back to fuzzy matching
        best_match = None
        best_ratio = 0
        
        # Use partial ratio matching with a 70% threshold
        for key in landmarks.keys():
            ratio = fuzz.partial_ratio(query, key)
            if ratio > best_ratio and ratio > 70:
                best_ratio = ratio
                best_match = landmarks[key]
        
        if best_match:
            return {"landmark": best_match, "is_fuzzy": True}
        return None

    def _handle_followup(self, user_input: str) -> Optional[str]:
        """
        Handle follow-up questions about the current landmark.
        Detects question type and generates appropriate detailed response.
        
        Special handling is implemented for the RPI Union with additional
        categories for events and facilities.
        
        Args:
            user_input: The user's follow-up question
            
        Returns:
            Optional[str]: Detailed response about requested aspect,
                          or None if question type not recognized
        """
        query = user_input.lower()
        info = self.knowledge['landmarks'].get(self.context['current_topic'], {})
        
        # Special handling for RPI Union queries
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
        """
        Generate detailed historical information about a landmark.
        
        Handles various historical data formats including:
        - Evolution over time
        - Original purpose
        - Builder information
        - Timeline of key events
        - Historical significance
        - Namesake information
        
        Args:
            info: Dictionary containing landmark information
            
        Returns:
            str: Formatted historical information
        """
        response_parts = []
        name = info.get('name', 'This landmark')
        
        if 'history' in info:
            history = info['history']
            # Handle various history data structures
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
            if 'namesake' in info:
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
        """
        Generate information about a landmark's architectural features.
        
        Includes:
        - Architectural style
        - Notable features
        - Architect information
        
        Args:
            info: Dictionary containing landmark information
            
        Returns:
            str: Formatted architectural information
        """
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
        """
        Generate information about a landmark's current usage.
        
        Handles both standard and RPI Union-specific information structures:
        - Departments housed
        - Available facilities
        - Student activities (Union-specific)
        - Regular events
        - Management information
        
        Args:
            info: Dictionary containing landmark information
            
        Returns:
            str: Formatted current use information
        """
        response_parts = []
        name = info.get('name', 'This landmark')
        
        # Handle standard current use information
        if 'current_use' in info:
            current = info['current_use']
            if isinstance(current, dict):
                if 'departments' in current:
                    response_parts.append(f"{name} currently houses {', '.join(current['departments'])}.")
                elif 'department' in current:
                    response_parts.append(f"{name} currently houses the {current['department']}.")
                if 'facilities' in current:
                    response_parts.append(f"Its facilities include: {', '.join(current['facilities'])}.")
        
        # Handle RPI Union specific features
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
        """
        Generate information specifically about RPI Union events and activities.
        
        Provides details about:
        - Regular events
        - Student clubs and activities
        - Types of supported activities
        
        Args:
            info: Dictionary containing Union information
            
        Returns:
            str: Formatted events and activities information
        """
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
        """
        Generate information about RPI Union facilities.
        
        Provides details about available spaces and facilities
        within the Union building.
        
        Args:
            info: Dictionary containing Union information
            
        Returns:
            str: Formatted facilities information
        """
        response_parts = []
        name = info.get('name', 'The Union')
        
        if 'features' in info and 'facilities' in info['features']:
            response_parts.append(f"{name}'s facilities include: {', '.join(info['features']['facilities'])}.")
        
        return " ".join(response_parts)

    def _analyze_noisy_input(self, user_input: str) -> Dict:
        """
        Use GPT to analyze ambiguous or unclear user input.
        
        Sends the input to OpenAI's GPT model for advanced natural language
        understanding and landmark identification. The model analyzes the input
        and attempts to match it to known landmarks, even when the reference
        is unclear or misspelled.
        
        Args:
            user_input: The user's original query text
            
        Returns:
            Dict: Analysis results containing:
                - landmark: Identified landmark key or null
                - original: Original user input
                - confidence: Confidence level (high/medium/low)
                - reasoning: Explanation of the match
        
        Raises:
            Exception: If GPT API call fails or returns invalid response
        """
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
                temperature=0.3  # Lower temperature for more consistent analysis
            )
            
            return json.loads(response.choices[0].message['content'])
        except Exception as e:
            logging.error(f"Error in GPT analysis: {e}")
            return {}