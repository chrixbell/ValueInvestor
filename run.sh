   source .venv/bin/activate
   export OPENAI_API_KEY="sk-5678ijklmnopabcd5678ijklmnopabcd5678ijkl"
   valueinvestor scan          # Full pipeline (fetch → screen → LLM analysis → report)
   valueinvestor screen        # Screening only (no API key needed)
   valueinvestor serve         # Web dashboard at localhost:8000

