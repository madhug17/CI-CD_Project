# ==========================================
# STEP 1: THE FOUNDATION (Operating System)
# ==========================================
# We pull a pre-built, ultra-lightweight Linux server that already has Python 3.10 installed.
# "slim" means it stripped out all the unnecessary junk so your file size stays small.
FROM python:3.10-slim

# ==========================================
# STEP 2: THE WORKSPACE (Setting the Room)
# ==========================================
# This creates a folder named /app inside the container and moves us inside it.
# Everything we do from here on out happens inside this specific folder.
WORKDIR /app

# ==========================================
# STEP 3: THE PACKING LIST (Dependencies)
# ==========================================
# We copy the requirements.txt file from your laptop into the container FIRST.
# We do this separately from the rest of the code to save build time later.
COPY requirements.txt .

# ==========================================
# STEP 4: THE INSTALLATION (Building the Engine)
# ==========================================
# This RUN command only executes ONCE while the container is being built.
# It reads your requirements.txt and installs FastAPI, Uvicorn, Streamlit, etc.
# The extra URL specifically forces the ultra-lightweight CPU version of PyTorch.
RUN pip install --no-cache-dir -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

# ==========================================
# STEP 5: MOVING THE FURNITURE (Your Code)
# ==========================================
# COPY [Everything on your laptop] to [The /app folder in the container]
# This grabs main.py, frontend.py, and your saved_model folder.
COPY . .

# ==========================================
# STEP 6: PUNCHING A HOLE IN THE WALL (Port)
# ==========================================
# Containers are locked down for security. This opens port 8000 so web traffic
# can actually reach your FastAPI server from the outside world.
EXPOSE 8000

# ==========================================
# STEP 7: TURNING THE KEY (Booting Up)
# ==========================================
# Unlike RUN, this command waits until the container actually turns on.
# When you type 'docker run' in your terminal, this is the command that fires to start your live server.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]