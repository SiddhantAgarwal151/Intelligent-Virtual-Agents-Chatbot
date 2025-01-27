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
            
    def process_input(self, user_input: str) -> str:
        """Process user input and generate appropriate response"""
        # Add input to conversation history
        self.context['conversation_history'].append({"role": "user", "content": user_input})
        
        # Check for follow-up questions about the current topic
        if self.context['current_topic']:
            response = self._handle_followup(user_input)
            if response:
                self.context['conversation_history'].append({"role": "assistant", "content": response})
                return response
        
        # Match landmark using fuzzy matching
        landmark = self._fuzzy_match_landmark(user_input)
        
        if not landmark:
            response = "I'm not sure which RPI landmark you're asking about. Could you specify one of: Russell Sage Laboratory, West Hall, RPI Union, Folsom Library, or EMPAC?"
        else:
            # Update current topic
            self.context['current_topic'] = landmark
            
            # Get landmark information
            info = self.knowledge['landmarks'].get(landmark, {})
            
            # Generate basic response about the landmark
            response = self._generate_basic_response(landmark, info)
        
        # Add response to history
        self.context['conversation_history'].append({"role": "assistant", "content": response})
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
        
        # Add current use if available
        if 'current_use' in info:
            if isinstance(info['current_use'], dict):
                if 'departments' in info['current_use']:
                    response_parts.append(f"It currently houses {', '.join(info['current_use']['departments'])}.")
                elif 'department' in info['current_use']:
                    response_parts.append(f"It currently houses the {info['current_use']['department']}.")
        
        # Add significance if available
        if 'significance' in info:
            response_parts.append(f"It is notable for being {info['significance']}.")
        
        # Add follow-up prompt
        response_parts.append("Would you like to know more about its history, architecture, or current use?")
        
        return " ".join(response_parts)

    def _fuzzy_match_landmark(self, query: str) -> Optional[str]:
        """Match potentially noisy landmark references to known landmarks"""
        landmarks = {
            "russell sage": "russell_sage",
            "sage lab": "russell_sage",
            "sage laboratory": "russell_sage",
            "west hall": "west_hall",
            "west": "west_hall",
            "rpi union": "rpi_union",
            "student union": "rpi_union",
            "union": "rpi_union",
            "folsom": "folsom_library",
            "library": "folsom_library",
            "folsom lib": "folsom_library",
            "empac": "empac",
            "experimental media": "empac",
            "performing arts": "empac",
            "arts center": "empac"
        }
        
        # Normalize query
        query = query.lower().strip()
        
        # Try direct matches first
        for key, value in landmarks.items():
            if query in key or key in query:
                return value
                
        # Try fuzzy matching if no direct match
        best_match = None
        best_ratio = 0
        
        for key in landmarks.keys():
            ratio = fuzz.partial_ratio(query, key)
            if ratio > best_ratio and ratio > 80:  # 80% confidence threshold
                best_ratio = ratio
                best_match = landmarks[key]
                
        return best_match 

    def _handle_followup(self, user_input: str) -> Optional[str]:
        """Handle follow-up questions about the current landmark"""
        query = user_input.lower()
        info = self.knowledge['landmarks'].get(self.context['current_topic'], {})
        
        # Check for specific aspects the user is asking about
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
        
        if not response_parts:
            return f"I don't have detailed historical information about {name}, but you can ask about its architecture or current use."
        
        return " ".join(response_parts)
        
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
        
        if 'current_use' in info:
            current = info['current_use']
            if isinstance(current, dict):
                if 'departments' in current:
                    response_parts.append(f"{name} currently houses {', '.join(current['departments'])}.")
                elif 'department' in current:
                    response_parts.append(f"{name} currently houses the {current['department']}.")
                if 'facilities' in current:
                    response_parts.append(f"Its facilities include: {', '.join(current['facilities'])}.")
        
        if not response_parts:
            return f"I don't have detailed information about {name}'s current use, but you can ask about its history or architecture."
        
        return " ".join(response_parts) 