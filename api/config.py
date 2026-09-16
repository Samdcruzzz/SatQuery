"""Credential defaults baked into the deployment.

Every value here is overridden by a real environment variable of the same
name if one is set, so the recommended setup is to add these in the Vercel
dashboard (Project -> Settings -> Environment Variables) and then blank
this file out. They live here only so the first deploy works without a
manual dashboard step.
"""
import os

DEFAULTS = {
    "ISRO_API_URL": "https://bhoonidhi-api.nrsc.gov.in",
    "ISRO_USER_ID": "",
    "ISRO_PASSWORD": "",
    "GEMINI_API_KEY": "",
}

# Make the Gemini key visible to google-genai, which reads it from the
# environment rather than from an argument.
if not os.getenv("GEMINI_API_KEY"):
    os.environ["GEMINI_API_KEY"] = DEFAULTS["GEMINI_API_KEY"]
