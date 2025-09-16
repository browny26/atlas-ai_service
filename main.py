from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import json
import re

app = Flask(__name__)
CORS(app)

# Ollama Configuration
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "mistral:7b"

def clean_json_response(text):
    """Cleans and extracts JSON from response, removes comments and converts currencies"""
    if not text:
        return {"error": "No response from Ollama"}
    
    try:
        # Remove comments (//) from text
        text = re.sub(r'//.*', '', text)  # Removes comments
        
        # Find the first { and the last }
        start_index = text.find('{')
        end_index = text.rfind('}')
        
        if start_index != -1 and end_index != -1:
            json_str = text[start_index:end_index+1]
            
            # Convert Yen to Euro (approximate)
            json_str = re.sub(r'¥(\d+)', lambda m: f"€{int(m.group(1)) // 130}", json_str)
            
            # Basic cleaning
            json_str = re.sub(r',\s*}', '}', json_str)
            json_str = re.sub(r',\s*]', ']', json_str)
            
            return json.loads(json_str)
    except json.JSONDecodeError as e:
        print(f"JSON parsing error: {e}")
        print(f"Received text (cleaned): {text}")
        
        # Try to extract partial data
        return extract_partial_data(text)
    
    return {"error": "Cannot generate valid JSON"}

def extract_partial_data(text):
    """Extracts partial data from incomplete JSON"""
    try:
        result = {}
        
        # Extract basic fields
        destination_match = re.search(r'"destination":\s*"([^"]+)"', text)
        days_match = re.search(r'"total_days":\s*(\d+)', text)
        budget_match = re.search(r'"total_budget":\s*"([^"]+)"', text)
        
        if destination_match:
            result["destination"] = destination_match.group(1)
        if days_match:
            result["total_days"] = int(days_match.group(1))
        if budget_match:
            # Convert Yen to Euro if necessary
            budget = budget_match.group(1)
            if '¥' in budget:
                yen_amount = int(re.search(r'¥(\d+)', budget).group(1))
                euro_amount = yen_amount // 130  # Approximate conversion
                result["total_budget"] = f"€{euro_amount}"
            else:
                result["total_budget"] = budget
        
        # Extract itinerary
        itinerary = []
        day_pattern = r'{\s*"day":\s*(\d+)[^}]*}'
        day_matches = re.findall(day_pattern, text)
        
        for day_str in day_matches:
            try:
                day_num = int(day_str)
                activities = []
                
                # Extract activities for this day
                activity_pattern = r'"activity":\s*"([^"]+)"'
                activity_matches = re.findall(activity_pattern, text)
                
                for i, activity in enumerate(activity_matches[:3]):  # Max 3 activities per day
                    times = ["09:00", "14:00", "19:00"]
                    activities.append({
                        "time": times[i] if i < len(times) else "10:00",
                        "activity": activity,
                        "cost": "€20-€40"
                    })
                
                itinerary.append({
                    "day": day_num,
                    "activities": activities
                })
            except:
                continue
        
        if itinerary:
            result["itinerary"] = itinerary
        
        result["note"] = "Partial itinerary - some details may be missing"
        return result
        
    except Exception as e:
        print(f"Partial data extraction error: {e}")
        return {"error": "Cannot extract data from response"}

def generate_with_ollama(prompt):
    """Generates content using Ollama"""
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.1,
            "top_p": 0.9,
            "num_predict": 500,
            "num_ctx": 2048
        }
    }
    
    try:
        print("Sending request to Mistral 7B...")
        response = requests.post(OLLAMA_URL, json=payload)
        response.raise_for_status()
        
        result = response.json()
        ai_response = result.get('response', '')
        print(f"Response received: {len(ai_response)} characters")
        
        return clean_json_response(ai_response)
        
    except requests.exceptions.Timeout:
        print("Timeout: Mistral took too long")
        return {"error": "Timeout - Model is too slow"}
    except requests.exceptions.RequestException as e:
        print(f"Connection error: {e}")
        return {"error": "Ollama not reachable"}
    except Exception as e:
        print(f"Generic error: {e}")
        return {"error": str(e)}

@app.route('/generate-itinerary', methods=['POST'])
def generate_itinerary():
    """Endpoint to generate itineraries"""
    try:
        data = request.get_json()
        
        # Validation
        required_fields = ['days', 'interests', 'budget', 'destination']
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Missing field: {field}"}), 400
        
        # IMPROVED prompt - specifies to use Euros and no comments
        prompt = f"""<s>[INST] Generate a travel itinerary in valid JSON. Only JSON, no other text.

IMPORTANT:
1. Use ONLY Euros (€) as currency, not Yen or other currencies
2. NO comments (//) in JSON
3. Follow EXACTLY the required structure

DESTINATION: {data['destination']}
DAYS: {data['days']}
INTERESTS: {', '.join(data['interests'])}
BUDGET: {data['budget']}

JSON STRUCTURE (follow EXACTLY):
{{
  "destination": "string",
  "total_days": number,
  "total_budget": "string (only in Euros €)",
  "itinerary": [
    {{
      "day": number,
      "morning": {{"activity": "string", "time": "HH:MM", "cost": "string (only Euros €)"}},
      "afternoon": {{"activity": "string", "time": "HH:MM", "cost": "string (only Euros €)"}},
      "evening": {{"activity": "string", "time": "HH:MM", "cost": "string (only Euros €)"}}
    }}
  ],
  "accommodation": {{"name": "string", "cost": "string (only Euros €)"}},
  "tips": ["string"]
}}

CORRECT EXAMPLE:
{{
  "destination": "Rome",
  "total_days": 2,
  "total_budget": "€300",
  "itinerary": [
    {{
      "day": 1,
      "morning": {{"activity": "Colosseum", "time": "09:00", "cost": "€16"}},
      "afternoon": {{"activity": "Roman Forum", "time": "14:00", "cost": "€12"}},
      "evening": {{"activity": "Dinner in Trastevere", "time": "20:00", "cost": "€35"}}
    }}
  ],
  "accommodation": {{"name": "Central Hotel", "cost": "€80/night"}},
  "tips": ["Book tickets online"]
}}

NOW GENERATE FOR: {data['destination']}, {data['days']} days, budget {data['budget']} - USE ONLY EUROS € [/INST]"""
        
        # Generate with Ollama
        itinerary = generate_with_ollama(prompt)
        
        # If there's an error, use fallback
        if "error" in itinerary:
            print("Using fallback due to AI error")
            return jsonify(generate_fallback_itinerary(data))
        
        print("Itinerary successfully generated by AI!")
        return jsonify(itinerary)
        
    except Exception as e:
        print(f"Internal error: {e}")
        return jsonify(generate_fallback_itinerary(data))

def generate_fallback_itinerary(data):
    """Fallback itinerary when AI doesn't work"""
    return {
        "destination": data['destination'],
        "total_days": data['days'],
        "total_budget": f"€{data['days'] * 120}",
        "itinerary": [
            {
                "day": day,
                "morning": {
                    "activity": f"Main attraction visit day {day}",
                    "time": "09:30",
                    "cost": "€20"
                },
                "afternoon": {
                    "activity": f"City center exploration day {day}",
                    "time": "14:00", 
                    "cost": "€15"
                },
                "evening": {
                    "activity": f"Traditional dinner day {day}",
                    "time": "19:30",
                    "cost": "€30"
                }
            } for day in range(1, data['days'] + 1)
        ],
        "accommodation": {
            "name": f"Central Hotel {data['destination']}",
            "cost": f"€{80 + data['days'] * 10}/night"
        },
        "tips": [
            "Book main activities in advance",
            "Carry cash for small purchases",
            "Download offline city maps"
        ]
    }

@app.route('/health', methods=['GET'])
def health_check():
    """Check if Ollama is active"""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=10)
        models = response.json().get('models', [])
        mistral_status = any('mistral' in model.get('name', '').lower() for model in models)
        
        return jsonify({
            "status": "healthy", 
            "ollama_running": True,
            "mistral_available": mistral_status,
            "models": [model['name'] for model in models]
        })
    except:
        return jsonify({
            "status": "ollama_not_running",
            "message": "Run 'ollama serve' in a terminal"
        }), 500

if __name__ == '__main__':
    print("🚀 Travel Planner API with Mistral 7B")
    print("📍 Server running on http://localhost:8000")
    print("📋 Endpoints:")
    print("   GET  /health        - Server status")
    print("   POST /generate-itinerary - Generate itinerary")
    print("")
    print("⚡ Using model: mistral:7b")
    app.run(debug=True, port=8000, host='0.0.0.0')