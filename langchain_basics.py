# %%
"""
LangChain Basics - Demonstrating core concepts and abstractions.
"""
import asyncio
import os
from typing import List, Dict, Any
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate,HumanMessagePromptTemplate

# Load environment variables
load_dotenv()
# %%
class LangChainBasicsDemo:
    """Demonstrates fundamental LangChain concepts and usage patterns."""
    def __init__(self):
        """Initialize the demo with different LLM providers."""
        self.models = {}
        
        # Initialize OpenAI model if API key is available
        if os.getenv( "OPENAI_API_KEY" ):
            self.models["openai"] = ChatOpenAI(
                model="gpt-4o",
                temperature=0.7,
                max_tokens=1000
            )
        
        # Initialize Anthropic model if API key is available
        if os.getenv("ANTHROPIC_API_KEY"):
            self.models["anthropic"] = ChatAnthropic(
                model="claude-3-haiku-20240307",
                temperature=0.7,
                max_tokens=1000
            )
    
    async def demonstrate_basic_chat(self):
        """Demonstrate basic chat functionality with language models."""
        print(" Basic Chat Demonstration")
        print("=" * 40)
        
        if not self.models:
            print("No models available for demonstration.")
            return
        
        # Create a simple message
        messages = [
            SystemMessage(content="You are a helpful customer service assistant."),
            HumanMessage(content="Hello, I need help with my account.")
        ]
        
        for provider_name, model in self.models.items():
            print(f"\n Testing {provider_name} Provider:")
            
            try:
                response = await model.ainvoke( messages )
                print(f"Response: {response.content[:200]}...")
            except Exception as e:
                print(f"Error: {str(e)}")
    
    async def demonstrate_prompt_templates(self):
        """Demonstrate the use of prompt templates for structured interactions."""
        print("\n Prompt Template Demonstration")
        print("=" * 40)
        
        if not self.models:
            print("No models available for demonstration.")
            return
        
        # Create a structured prompt template
        system_template = """
        You are a customer service agent for{company_name}.
        Your role is to {agent_role}. 
        Always maintain a {tone} tone and provide {response_style} responses.
        """
        
        human_template = """
        Customer inquiry: {customer_message}
        Customer context: {customer_context}
        Please provide an appropriate response.
        """
        
        # Build the chat prompt template
        chat_prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(system_template),
            HumanMessagePromptTemplate.from_template(human_template)
        ])
        
        # Example data for template formatting
        template_data = {
            "company_name": "TechCorp Solutions",
            "agent_role": "help customers with technical issues",
            "tone": "professional and empathetic",
            "response_style": "clear and actionable",
            "customer_message": "My application keeps crashing when I try to save files.",
            "customer_context": "Premium customer, using version 2.1.3 on Windows 11"
        }
        
        
        # Format the prompt with the data
        formatted_messages = chat_prompt.format_messages(**template_data)
        
        print(" Formatted Prompt:")
        for message in formatted_messages:
            print(f"{message.__class__.__name__}: {message.content[:150]}...")
        
        # Test with available models
        for provider_name, model in self.models.items():
            print(f"\n {provider_name} Response:")
            
            try:
                response = await model.ainvoke( formatted_messages )
                print(f"Response: {response.content[:300]}...")
            except Exception as e:
                print(f"Error: {str(e)}")
                
    async def demonstrate_conversation_memory(self):
        """Demonstrate how to maintain conversation context across multiple interactions."""
        print("\n Conversation Memory Demonstration")
        print("=" * 40)
        
        if not self.models:
            print("No models available for demonstration.")
            return
        
        # Use the first available model
        model = next(iter(self.models.values()))
        
        # Simulate a multi-turn conversation
        conversation_history = [
            SystemMessage(content="You are a helpful customer service assistant.\
                Remember the context of our conversation.")
        ]
        
        customer_messages = [
            "Hi, I'm having trouble with my account login.",
            "I've tried resetting my password but it's not working.",
            "What other options do I have to regain access?"
        ]
        
        for i, customer_message in enumerate(customer_messages, 1):
            print(f"\n️  Turn {i}:")
            print(f"Customer: {customer_message}")
            
            # Add customer message to conversation history
            conversation_history.append(HumanMessage(content=customer_message))
            
            try:
                response = await model.ainvoke( conversation_history )
                print(f"Assistant: {response.content[:200]}...")
                
                # Add AI response to conversation history
                conversation_history.append(AIMessage(content=response.content))
            except Exception as e:
                print(f"Error: {str(e)}")
                break
            
    async def demonstrate_structured_output(self):
        """Demonstrate how to get structured output from language models."""
        print("\n Structured Output Demonstration")
        print("=" * 40)
        
        if not self.models:
            print("No models available for demonstration.")
            return
        
        # Create a prompt that requests structured output
        system_message = """
        You are a customer inquiry analyzer.
        Analyze the customer message and respond with a JSON object containing:
        - category: the type of inquiry (technical, billing, general, complaint)
        - urgency: urgency level (low, medium, high, critical)
        - sentiment: customer sentiment (positive, neutral, negative)
        - confidence: your confidence in the analysis (0.0 to 1.0)
        
        Respond only with valid JSON, no additional text. Do not qualify the response with json tag.
        """
        
        human_message = """
        I'm really frustrated! My premium subscription was charged twice this month 
        and I can't get through to anyone for help. This is unacceptable!
        """
        
        structured_prompt = ChatPromptTemplate.from_messages([
            SystemMessage( content=system_message ),
            HumanMessage( content=human_message )
        ])
        
        model = next(iter(self.models.values()))
        
        try:
            response = await model.ainvoke( structured_prompt.format_messages() )
            print(" Structured Analysis:")
            print(response.content)
            
            # Attempt to parse as JSON
            import json
            
            try:
                parsed_response = json.loads(response.content)
                print("\n✅ Successfully parsed JSON:")
                for key, value in parsed_response.items():
                    print(f" {key}: {value}")
            except json.JSONDecodeError:
                print("⚠️ Response is not valid JSON")
        except Exception as e:
            print(f"Error: {str(e)}")
            
# %%
async def main():
    """Run the LangChain basics demonstration."""
    demo = LangChainBasicsDemo()
    
    if not demo.models:
        print("❌ No LLM providers configured. Please set up API keys in your .env file.")
        return
    
    print(" LangChain Fundamentals Demonstration")
    print("=" * 50)
    
    await demo.demonstrate_basic_chat()
    await demo.demonstrate_prompt_templates()
    await demo.demonstrate_conversation_memory()
    await demo.demonstrate_structured_output()
    
    print("\n LangChain basics demonstration complete!")
# %%
if __name__ == "__main__":
    asyncio.run(main())
# %%
