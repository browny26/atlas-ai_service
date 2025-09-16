from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import json
import re

app = Flask(__name__)
CORS(app)

# Configurazione Ollama
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "mistral:7b"

def clean_json_response(text):
    """Pulisce e estrae JSON dalla risposta, rimuove commenti e converte valute"""
    if not text:
        return {"error": "Nessuna risposta da Ollama"}
    
    try:
        # Rimuovi commenti (//) dal testo
        text = re.sub(r'//.*', '', text)  # Rimuove i commenti
        
        # Cerca il primo { e l'ultimo }
        start_index = text.find('{')
        end_index = text.rfind('}')
        
        if start_index != -1 and end_index != -1:
            json_str = text[start_index:end_index+1]
            
            # Converti Yen in Euro (approssimativo)
            json_str = re.sub(r'¥(\d+)', lambda m: f"€{int(m.group(1)) // 130}", json_str)
            
            # Pulizia base
            json_str = re.sub(r',\s*}', '}', json_str)
            json_str = re.sub(r',\s*]', ']', json_str)
            
            return json.loads(json_str)
    except json.JSONDecodeError as e:
        print(f"Errore parsing JSON: {e}")
        print(f"Testo ricevuto (pulito): {text}")
        
        # Prova a estrarre dati parziali
        return extract_partial_data(text)
    
    return {"error": "Impossibile generare JSON valido"}

def extract_partial_data(text):
    """Estrae dati parziali dal JSON incompleto"""
    try:
        result = {}
        
        # Estrai campi base
        destination_match = re.search(r'"destination":\s*"([^"]+)"', text)
        days_match = re.search(r'"total_days":\s*(\d+)', text)
        budget_match = re.search(r'"total_budget":\s*"([^"]+)"', text)
        
        if destination_match:
            result["destination"] = destination_match.group(1)
        if days_match:
            result["total_days"] = int(days_match.group(1))
        if budget_match:
            # Converti Yen in Euro se necessario
            budget = budget_match.group(1)
            if '¥' in budget:
                yen_amount = int(re.search(r'¥(\d+)', budget).group(1))
                euro_amount = yen_amount // 130  # Conversione approssimativa
                result["total_budget"] = f"€{euro_amount}"
            else:
                result["total_budget"] = budget
        
        # Estrai itinerario
        itinerary = []
        day_pattern = r'{\s*"day":\s*(\d+)[^}]*}'
        day_matches = re.findall(day_pattern, text)
        
        for day_str in day_matches:
            try:
                day_num = int(day_str)
                activities = []
                
                # Estrai attività per questo giorno
                activity_pattern = r'"activity":\s*"([^"]+)"'
                activity_matches = re.findall(activity_pattern, text)
                
                for i, activity in enumerate(activity_matches[:3]):  # Max 3 attività per giorno
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
        
        result["note"] = "Itinerario parziale - alcuni dettagli potrebbero mancare"
        return result
        
    except Exception as e:
        print(f"Errore estrazione dati parziali: {e}")
        return {"error": "Impossibile estrarre dati dalla risposta"}

def generate_with_ollama(prompt):
    """Genera contenuto usando Ollama"""
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
        print("Invio richiesta a Mistral 7B...")
        response = requests.post(OLLAMA_URL, json=payload)
        response.raise_for_status()
        
        result = response.json()
        ai_response = result.get('response', '')
        print(f"Risposta ricevuta: {len(ai_response)} caratteri")
        
        return clean_json_response(ai_response)
        
    except requests.exceptions.Timeout:
        print("Timeout: Mistral ha impiegato troppo tempo")
        return {"error": "Timeout - Il modello è troppo lento"}
    except requests.exceptions.RequestException as e:
        print(f"Errore connessione: {e}")
        return {"error": "Ollama non raggiungibile"}
    except Exception as e:
        print(f"Errore generico: {e}")
        return {"error": str(e)}

@app.route('/generate-itinerary', methods=['POST'])
def generate_itinerary():
    """Endpoint per generare itinerari"""
    try:
        data = request.get_json()
        
        # Validazione
        required_fields = ['days', 'interests', 'budget', 'destination']
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Campo mancante: {field}"}), 400
        
        # Prompt MIGLIORATO - specifica di usare Euro e niente commenti
        prompt = f"""<s>[INST] Genera un itinerario di viaggio in JSON valido. Solo JSON, nessun altro testo.

IMPORTANTE:
1. Usa SOLO Euro (€) come valuta, non Yen o altre valute
2. NO commenti (//) nel JSON
3. Segui ESATTAMENTE la struttura richiesta

DESTINAZIONE: {data['destination']}
GIORNI: {data['days']}
INTERESSI: {', '.join(data['interests'])}
BUDGET: {data['budget']}

STRUTTURA JSON (segui ESATTAMENTE):
{{
  "destination": "string",
  "total_days": number,
  "total_budget": "string (solo in Euro €)",
  "itinerary": [
    {{
      "day": number,
      "morning": {{"activity": "string", "time": "HH:MM", "cost": "string (solo Euro €)"}},
      "afternoon": {{"activity": "string", "time": "HH:MM", "cost": "string (solo Euro €)"}},
      "evening": {{"activity": "string", "time": "HH:MM", "cost": "string (solo Euro €)"}}
    }}
  ],
  "accommodation": {{"name": "string", "cost": "string (solo Euro €)"}},
  "tips": ["string"]
}}

Esempio CORRETTO:
{{
  "destination": "Roma",
  "total_days": 2,
  "total_budget": "€300",
  "itinerary": [
    {{
      "day": 1,
      "morning": {{"activity": "Colosseo", "time": "09:00", "cost": "€16"}},
      "afternoon": {{"activity": "Foro Romano", "time": "14:00", "cost": "€12"}},
      "evening": {{"activity": "Cena Trastevere", "time": "20:00", "cost": "€35"}}
    }}
  ],
  "accommodation": {{"name": "Hotel Centro", "cost": "€80/notte"}},
  "tips": ["Prenotare biglietti online"]
}}

ORA GENERA PER: {data['destination']}, {data['days']} giorni, budget {data['budget']} - USA SOLO EURO € [/INST]"""
        
        # Genera con Ollama
        itinerary = generate_with_ollama(prompt)
        
        # Se c'è errore, fallback
        if "error" in itinerary:
            print("Usando fallback a causa di errore AI")
            return jsonify(generate_fallback_itinerary(data))
        
        print("Itinerario generato con successo dall'AI!")
        return jsonify(itinerary)
        
    except Exception as e:
        print(f"Errore interno: {e}")
        return jsonify(generate_fallback_itinerary(data))

def generate_fallback_itinerary(data):
    """Itinerario di fallback quando l'AI non funziona"""
    return {
        "destination": data['destination'],
        "total_days": data['days'],
        "total_budget": f"€{data['days'] * 120}",
        "itinerary": [
            {
                "day": day,
                "morning": {
                    "activity": f"Visita attrazione principale giorno {day}",
                    "time": "09:30",
                    "cost": "€20"
                },
                "afternoon": {
                    "activity": f"Esplorazione zona centrale giorno {day}",
                    "time": "14:00", 
                    "cost": "€15"
                },
                "evening": {
                    "activity": f"Cena tipica giorno {day}",
                    "time": "19:30",
                    "cost": "€30"
                }
            } for day in range(1, data['days'] + 1)
        ],
        "accommodation": {
            "name": f"Hotel Centrale {data['destination']}",
            "cost": f"€{80 + data['days'] * 10}/notte"
        },
        "tips": [
            "Prenotare in anticipo le attività principali",
            "Portare contanti per piccoli acquisti",
            "Scaricare mappe offline della città"
        ]
    }

@app.route('/health', methods=['GET'])
def health_check():
    """Verifica che Ollama sia attivo"""
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
            "message": "Esegui 'ollama serve' in un terminale"
        }), 500

if __name__ == '__main__':
    print("🚀 API Travel Planner con Mistral 7B")
    print("📍 Server in esecuzione su http://localhost:8000")
    print("📋 Endpoints:")
    print("   GET  /health        - Stato del server")
    print("   POST /generate-itinerary - Genera itinerario")
    print("")
    print("⚡ Usando modello: mistral:7b")
    app.run(debug=True, port=8000, host='0.0.0.0')