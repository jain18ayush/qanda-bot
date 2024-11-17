import discord
from discord.ext import commands
import chromadb
from transformers import AutoModel, AutoTokenizer  # For embeddings
import torch
from dotenv import load_dotenv
import os 

load_dotenv() 

# Initialize bot
intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix="!", intents=intents)

# Initialize ChromaDB client
client = chromadb.Client()

# Create a collection for storing the embeddings
vectorstore = client.create_collection(name="discord_embeddings")

# Initialize embedding model
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/paraphrase-MiniLM-L6-v2")
model = AutoModel.from_pretrained("sentence-transformers/paraphrase-MiniLM-L6-v2")

CHANNEL_ID_1 = "1307499332366503950"  # Replace with the actual channel ID
CHANNEL_ID_2 = "1307499419570274404"  # Replace with the actual channel ID
CHANNEL_ID_3 = "1307499443075022940"  # Replace with the actual channel ID

# Function to embed a message
def embed_text(text):
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        embeddings = model(**inputs).last_hidden_state.mean(dim=1)
    return embeddings.squeeze().cpu().numpy()  # Convert to numpy array for Chroma compatibility

# Ingest historical messages
@bot.event
async def on_ready():
    print(f"Logged in as {bot.user}")
    for channel_id in [CHANNEL_ID_1, CHANNEL_ID_2, CHANNEL_ID_3]:  # replace with actual IDs
        channel = bot.get_channel(channel_id)
        print("channel", channel)
        async for message in channel.history(limit=1000):  # Adjust limit as needed
            embedding = embed_text(message.content)
            vectorstore.add(
                documents=[message.content],
                embeddings=[embedding],
                metadatas=[{"message_id": message.id}],  # Optionally store metadata
                ids=[str(message.id)]  # Unique ID for each message
            )

# Update vectorstore when a new message is posted
@bot.event
async def on_message(message):
    if message.channel.id in [CHANNEL_ID_1, CHANNEL_ID_2, CHANNEL_ID_3]:
        embedding = embed_text(message.content)
        vectorstore.add(
            documents=[message.content],
            embeddings=[embedding],
            metadatas=[{"message_id": message.id}],
            ids=[str(message.id)]
        )

    await bot.process_commands(message)

# Query handling
@bot.command()
async def ask(ctx, *, query):
    query_embedding = embed_text(query)
    
    # Search for similar messages in the vectorstore
    results = vectorstore.query(
        query_embeddings=[query_embedding],
        n_results=5  # Adjust the number of top results
    )
    
    # Extract the top results and generate a response
    relevant_messages = [result['document'] for result in results['documents']]
    response = generate_response(query, relevant_messages)  # Use an LLM to generate the response

    await ctx.send(response)

def generate_response(query, relevant_messages):
    # Example function to generate a response
    # You could use GPT or any other model here to generate a response
    return f"Relevant messages: \n" + "\n".join(relevant_messages)

token = os.getenv("DISCORD_BOT_TOKEN")
bot.run(token)