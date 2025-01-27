from chatbot import RPIChatbot
import logging
import os
from dotenv import load_dotenv

def main():
    # Load environment variables
    load_dotenv()
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Initialize chatbot
    try:
        chatbot = RPIChatbot()
    except Exception as e:
        logging.error(f"Failed to initialize chatbot: {e}")
        return
        
    # Welcome message
    print("\nWelcome to the RPI History Chatbot!")
    print("Ask me anything about RPI's Landmarks.")
    print("Type 'quit' to exit.")
    
    # Main conversation loop
    while True:
        try:
            # Get user input
            user_input = input("\nYou: ").strip()
            
            # Check for exit command
            if user_input.lower() in ['quit', 'exit', 'bye']:
                print("\nThank you for chatting about RPI history! Goodbye!")
                break
                
            # Process input and get response
            response = chatbot.process_input(user_input)
            print("\nBot:", response)
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            logging.error(f"Error in conversation loop: {e}")
            print("\nI apologize, but I encountered an error. Please try again.")

if __name__ == "__main__":
    main() 