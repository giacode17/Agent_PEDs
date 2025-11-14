#!/usr/bin/env python3
"""
Local runner for the Pediatric Post-Discharge Agent with MLflow tracking.

This script allows you to chat with the agent locally and automatically
tracks all conversations in MLflow.

Usage:
    poetry run python run_local.py
"""
import os
from dotenv import load_dotenv
from peds_post_discharge_agent.agent import PediatricAgentService

# Load environment variables
load_dotenv()

def main():
    print("=" * 70)
    print("  Pediatric Post-Discharge Agent - Local Mode")
    print("=" * 70)
    print()
    print("MLflow tracking enabled: Conversations will be logged to ./mlruns")
    print("View metrics with: mlflow ui --backend-store-uri file:./mlruns")
    print()
    print("Type 'quit' or 'exit' to stop")
    print("=" * 70)
    print()

    # Create a mock context (for local use, not deployed)
    class MockContext:
        pass

    context = MockContext()

    # Configure with MLflow enabled
    params = {
        "space_id": os.getenv("SPACE_ID"),
        "mlflow_enabled": True,  # Enable MLflow tracking
        "mlflow_tracking_uri": "file:./mlruns",
        "mlflow_experiment_name": "peds_post_discharge_agent",
    }

    # Create the agent service
    agent = PediatricAgentService(context, params=params)

    print("Agent ready! Try asking:")
    print('  - "My child has a fever of 38.5°C and mild pain"')
    print('  - "Remind me to give Zyrtec every 12 hours"')
    print('  - "What foods are okay after tonsillectomy?"')
    print()

    # Interactive conversation loop
    while True:
        try:
            # Get user input
            user_input = input("You: ").strip()

            if not user_input:
                continue

            if user_input.lower() in ['quit', 'exit', 'bye']:
                print("\nGoodbye! Check your MLflow logs for conversation metrics.")
                break

            # Create request context
            request_ctx = {
                "input": {
                    "messages": [
                        {
                            "role": "user",
                            "content": user_input
                        }
                    ]
                }
            }

            # Get response from agent
            print("\nAgent: ", end="", flush=True)
            response = agent.generate(request_ctx)

            # Extract and display the response
            if "choices" in response and len(response["choices"]) > 0:
                message = response["choices"][0]["message"]
                content = message.get("content", "")
                print(content)
            else:
                print("(No response generated)")

            print()  # Add blank line for readability

        except KeyboardInterrupt:
            print("\n\nInterrupted! Goodbye.")
            break
        except Exception as e:
            print(f"\nError: {e}")
            print("Try another question or type 'quit' to exit.\n")

if __name__ == "__main__":
    main()
