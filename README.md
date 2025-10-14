## 🤖 Legacy AI Module

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)
![Ollama](https://img.shields.io/badge/Ollama-Mistral-orange?logo=ollama)
![Flask](https://img.shields.io/badge/Flask-2.3-lightgrey?logo=flask)

The **legacy AI module** of Atlas was built with **Python** and used **Ollama Mistral** to generate intelligent travel suggestions.  
The module exposes **REST endpoints using Flask**, which are called by the Java/Spring backend to provide AI-powered itinerary recommendations. 

Key points of the legacy AI module:

- Ran **locally** and implemented in **Python** with **Ollama Mistral**
- Exposes **REST endpoints via Flask** for backend integration  
- Generated **personalized travel suggestions** based on user preferences
- Was integrated with the backend for testing AI-assisted itinerary planning
- Deprecated in favor of **Groq + OpenAI** for scalability, cloud deployment, and more advanced AI features

This module remains part of the project history as a reference for the initial AI implementation.
