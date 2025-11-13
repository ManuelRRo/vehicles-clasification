import requests
import json

def askQuestion(question: str) -> str:
    data={
        "model": "gemma3",
        "messages": [
            {"role": "user", 
            "content": question}],
        "stream": False
    }
    url = "http://192.168.31.145:11434/api/chat"

    response = requests.post(url,json=data)

    response_json = json.loads(response.text)

    ai_reply = response_json["message"]["content"]


    return ai_reply