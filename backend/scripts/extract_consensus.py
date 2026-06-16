#!/usr/bin/env python3
import os
import json
import sys
import glob

def main():
    # Base dir: backend/uploads/simulations
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'uploads', 'simulations'))
    
    if not os.path.exists(base_dir):
        print(f"Simulation directory not found: {base_dir}")
        return
        
    # Find latest simulation directory
    sim_dirs = glob.glob(os.path.join(base_dir, '*'))
    sim_dirs = [d for d in sim_dirs if os.path.isdir(d)]
    
    if not sim_dirs:
        print("No simulation results found.")
        return
        
    latest_sim_dir = max(sim_dirs, key=os.path.getmtime)
    print(f"Extracting consensus from latest simulation: {os.path.basename(latest_sim_dir)}")
    
    # Read action logs
    actions = []
    for platform in ["twitter", "reddit"]:
        log_path = os.path.join(latest_sim_dir, platform, "actions.jsonl")
        if os.path.exists(log_path):
            with open(log_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        try:
                            # Parse JSONL and filter for interesting content
                            data = json.loads(line)
                            if 'action_type' in data and data['action_type'] in ['CREATE_POST', 'CREATE_COMMENT']:
                                agent_name = data.get('agent_name', 'Unknown Agent')
                                content = data.get('action_args', {}).get('content', '')
                                if content:
                                    actions.append(f"[{agent_name}]: {content}")
                        except Exception:
                            pass
                            
    if not actions:
        print("No debate actions found in the simulation logs.")
        return
        
    debate_transcript = "\n".join(actions[-200:]) # Limit to last 200 actions to avoid context explosion
    
    # Initialize LLM Client
    # We will use OpenAI client configured for LM Studio local inference
    try:
        from openai import OpenAI
    except ImportError:
        print("Please install openai: pip install openai")
        return
        
    # Default LM Studio settings
    client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")
    
    print("Synthesizing Final 16-Point Prediction Report...")
    try:
        response = client.chat.completions.create(
            model="local-model", # LM Studio intercepts this
            messages=[
                {"role": "system", "content": "You are a Master Quantitative AI. Your job is to read the attached simulation transcript of various market agents debating. You must distill their collective arguments, heavily weighing the hard mathematical data they reference, and output ONLY a valid 16-point JSON object mapping to the JSE prediction requirements. Do NOT output markdown, ONLY pure JSON."},
                {"role": "user", "content": f"Here is the finalized transcript of the forward-projection simulation:\n\n{debate_transcript}\n\nBased on this debate, synthesize the final market outlook into a structured JSON string. Output ONLY the JSON."}
            ],
            temperature=0.2, # Low temperature for strict analytical formatting
        )
        
        final_json_str = response.choices[0].message.content.strip()
        
        # Clean markdown wrappers if hallucinated
        if final_json_str.startswith("```json"):
            final_json_str = final_json_str[7:]
        if final_json_str.endswith("```"):
            final_json_str = final_json_str[:-3]
            
        final_json_str = final_json_str.strip()
        
        # Save output
        output_path = os.path.join(latest_sim_dir, "prediction_consensus.json")
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(final_json_str)
            
        print(f"✅ Success! 16-Point Consensus Prediction saved to: {output_path}")
        
    except Exception as e:
        print(f"Error calling local LLM for consensus extraction: {e}")

if __name__ == "__main__":
    main()
