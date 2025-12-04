import requests
import json
import argparse
import sys
import time
from typing import List, Dict, Any

class CausalClient:
    def __init__(self, base_url: str = "http://127.0.0.1:8000"):
        self.base_url = base_url
        self.chat_endpoint = f"{base_url}/chat"

    def query(self, user_query: str, history: List[Dict] = None, model_type: str = "graph") -> None:
        """
        Sends a query to the causal engine and streams the response.
        """
        if history is None:
            history = []

        payload = {
            "query": user_query,
            "history": history,
            "model_type": model_type
        }

        print(f"\n{'='*50}")
        print(f"QUERY: {user_query}")
        print(f"STRATEGY: {model_type}")
        print(f"{'='*50}\n")

        try:
            with requests.post(self.chat_endpoint, json=payload, stream=True) as response:
                response.raise_for_status()
                
                print("--- STREAMING RESPONSE ---\n")
                
                for line in response.iter_lines():
                    if not line:
                        continue
                        
                    decoded_line = line.decode('utf-8')
                    if decoded_line.startswith("data: "):
                        data_str = decoded_line[6:]
                        try:
                            event_data = json.loads(data_str)
                            self._handle_event(event_data)
                        except json.JSONDecodeError:
                            print(f"[WARN] Could not decode JSON: {data_str}")
                            
        except requests.exceptions.RequestException as e:
            print(f"\n[ERROR] Request failed: {e}")
        except KeyboardInterrupt:
            print("\n[INFO] Interrupted by user.")

    def _handle_event(self, event: Dict[str, Any]) -> None:
        """
        Pretty prints events from the server.
        """
        evt_type = event.get("event")
        data = event.get("data")
        
        if evt_type == "token":
            print(data, end="", flush=True)
        elif evt_type == "status":
            print(f"\n[STATUS]: {data}")
        elif evt_type == "concepts":
            print(f"[CONCEPTS]: {data}")
        elif evt_type == "sources":
            ids = event.get("ids", [])
            print(f"\n[SOURCES]: Found {len(ids)} relevant calls.")
            if ids:
                print(f"          IDs: {', '.join(ids[:5])}...")
        elif evt_type == "error":
            print(f"\n[SERVER ERROR]: {event.get('message')}")
        else:
            print(f"\n[EVENT]: {evt_type} -> {data}")

def main():
    parser = argparse.ArgumentParser(description="Test Client for Causal Engine")
    parser.add_argument("query", nargs="?", help="The query to send", default="Why are customers churning in Telecom?")
    parser.add_argument("--model", choices=["graph", "filter"], default="graph", help="Strategy to use")
    parser.add_argument("--multi", action="store_true", help="Run a suite of test queries")
    
    args = parser.parse_args()
    
    client = CausalClient()
    
    if args.multi:
        test_queries = [
            "Why are customers churning in Telecom?",
            "What are the common billing issues?",
            "How do agents handle angry customers?"
        ]
        for q in test_queries:
            client.query(q, model_type=args.model)
            time.sleep(1)
    else:
        client.query(args.query, model_type=args.model)

if __name__ == "__main__":
    main()
