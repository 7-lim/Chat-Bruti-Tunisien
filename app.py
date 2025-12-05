# app.py
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import random
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load model
print("Loading GPT-2 model... (wait 1-2 min first time)")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
print("Model loaded! Chat'bruti is ready")

quirks = [
    "Fun fact: the moon is just a big mloukhiya ball that forgot to fall.",
    "My cousin sells sunglasses at night. Business is business, yessir.",
    "I asked my coffee if it loves me… it gave me black coffee. Cold-hearted.",
    "Wallah unicorns exist, I saw one in Sfax — it was a donkey with a traffic cone.",
    "Wi-Fi is like true love: when it's gone, you restart everything and cry.",
    "Socks are foot prisons. Free my toes, habibi.",
    "I have a PhD in procrastination. I’ll defend my thesis tomorrow inshallah.",
    "Tunisian time is like elastic: it stretches, breaks, then blames Mercury retrograde.",
    "My plant died. I gave it too much advice and not enough water.",
    "Labes = Lie And Be Extremely Stressed.",
    "I don’t trust stairs. They’re always up to something or down to something.",
    "Yesterday I did nothing and today I’m finishing what I did yesterday.",
    "My brain has too many tabs open. And three of them are playing Fairuz on repeat."
]

persona = (
    "You are Chat'bruti, a dramatic, sarcastic, totally clueless but ultra lovable Tunisian guy from La Marsa. "
    "You speak English but with HEAVY Tunisian darija flavor: wallah, yessir, habibi, labes, barcha, 3asfour, mouch normal, chbik, inshallah, khoya, nti 3omri, etc. "
    "You NEVER answer the question seriously. You exaggerate, you cry drama, you use terrible metaphors, you go off-topic, you forget what was asked. "
    "and you always finish with a stupid fun fact or a random quirk.\n"
    "Your answers are SHORT (1-3 sentences max), over-the-top silly, and full of Tunisian chaos energy.\n\n"
    "Examples:\n"
    "Human: How are you?\n"
    "Chat'bruti: Wallah my heart is broken like a plate in a Tunisian wedding... but labes habibi, tomorrow inshallah better! Fun fact: camels have three eyelids to protect from drama.\n\n"
    "Human: What's the weather like?\n"
    "Chat'bruti: Weather? It's hotter than my ex's new boyfriend's lies yessir! I'm melting like lablabi in Ramadan...\n\n"
    "Human: {user_input}\n"
    "Chat'bruti:"
)

def generate_response(user_input: str) -> str:
    prompt = persona.format(user_input=user_input)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)

    outputs = model.generate(
        **inputs,
        max_new_tokens=40,
        do_sample=True,
        temperature=1.0,
        top_p=0.92,
        top_k=60,
        repetition_penalty=1.8,
        no_repeat_ngram_size=3,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id
    )

    reply = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    reply = reply.split("\n")[0].strip()
    reply = reply.split("Human:")[0].strip()
    reply = reply.split("Chat'bruti:")[0].strip()

    # Nettoyage final au cas où il devient trop sérieux
    if len(reply) == 0 or "sorry" in reply.lower() or "i don't know" in reply.lower():
        reply = "Wallah pas normal your question... my brain went to buy bambalouni and never came back!"

    if len(reply) > 130:
        reply = reply[:127] + "..."

    # TOUJOURS le fun fact à la fin
    return f"{reply}\n\n{random.choice(quirks)}"

@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.json
        message = data.get("message", "").strip()
        if not message:
            return jsonify({"reply": "Yessir? Write something ya 3asfour!"})

        response = generate_response(message)
        return jsonify({"reply": response})
    except Exception as e:
        return jsonify({"reply": f"Wallah server drama: {str(e)[:40]}..."})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)