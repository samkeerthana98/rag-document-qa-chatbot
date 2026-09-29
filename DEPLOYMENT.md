# AWS EC2 Deployment Guide

This guide covers deploying the RAG Document Q&A Chatbot on a single AWS EC2
instance. The setup runs Ollama (LLM) directly on the instance and the Flask
application inside a Docker container. No paid LLM APIs, no Kubernetes, no ECS.

---

## Architecture

```
EC2 Instance (t3.large or t3.xlarge)
├── Ollama (host process, port 11434)
│   └── llama3.2 model weights (~2 GB)
└── Docker container: rag-chatbot (port 5000)
    ├── Flask + Gunicorn
    ├── HuggingFace sentence-transformers (all-MiniLM-L6-v2)
    └── ChromaDB (data persisted on host via volume)
```

The container talks to Ollama via `http://host.docker.internal:11434`. Ollama
runs directly on the EC2 host; the container reaches it using the
`host.docker.internal` hostname, which `--add-host` maps to the host gateway IP.

---

## 1. EC2 Instance Requirements

### Instance type

| Minimum | Recommended |
|---|---|
| `t3.large` (8 GB RAM, 2 vCPU) | `t3.xlarge` (16 GB RAM, 4 vCPU) |

Memory breakdown on `t3.large`:
- llama3.2 model in RAM: ~4 GB
- sentence-transformers (all-MiniLM-L6-v2): ~1 GB
- OS + Docker overhead: ~1.5 GB
- Leaves ~1.5 GB headroom (tight but workable)

Use `t3.xlarge` if you expect concurrent users or want comfortable headroom.

### AMI

Use **Amazon Linux 2023** (AL2023) or **Ubuntu 22.04 LTS**. Both are tested
with Docker and Ollama. The commands below use Amazon Linux 2023 syntax; see
inline notes for Ubuntu differences.

### Storage

- Root EBS volume: **30 GB minimum** (Docker image is ~10 GB, Ollama model
  cache is ~4 GB, leave room for logs and uploads).
- Recommended: **40 GB** gp3.

### SSH key pair

Create or select an existing key pair when launching the instance. You will
need it to SSH in.

---

## 2. Security Group

Open the following inbound rules. All outbound traffic can remain open (the
default).

| Type | Protocol | Port | Source | Purpose |
|---|---|---|---|---|
| SSH | TCP | 22 | Your IP only | Remote management |
| Custom TCP | TCP | 5000 | 0.0.0.0/0 | Flask application |

Do **not** expose port 11434 (Ollama) to the internet — it has no
authentication and would allow anyone to query your LLM for free.

For a production setup you would put the app behind a reverse proxy (nginx or
ALB) on port 80/443 and remove the direct port-5000 rule. For this portfolio
project, direct access on port 5000 is sufficient.

---

## 3. Connect to the Instance

```bash
ssh -i /path/to/your-key.pem ec2-user@<your-instance-public-ip>
```

On Ubuntu the default user is `ubuntu`, not `ec2-user`:

```bash
ssh -i /path/to/your-key.pem ubuntu@<your-instance-public-ip>
```

---

## 4. Install Docker

### Amazon Linux 2023

```bash
# Update packages
sudo dnf update -y

# Install Docker
sudo dnf install -y docker

# Start Docker and enable it on reboot
sudo systemctl start docker
sudo systemctl enable docker

# Add ec2-user to the docker group so you can run docker without sudo
sudo usermod -aG docker ec2-user

# Log out and back in for the group change to take effect
exit
# SSH back in
```

### Ubuntu 22.04

```bash
sudo apt-get update
sudo apt-get install -y ca-certificates curl
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg \
  | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
  https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" \
  | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io
sudo systemctl start docker
sudo systemctl enable docker
sudo usermod -aG docker ubuntu
exit  # log out and back in
```

### Verify

```bash
docker --version
# Docker version 24.x.x, build ...
```

---

## 5. Install Ollama

Ollama runs directly on the EC2 host (not in Docker). The installer script
works on both Amazon Linux 2023 and Ubuntu.

```bash
# Download and run the official installer
curl -fsSL https://ollama.com/install.sh | sh

# The installer registers an ollama systemd service automatically.
# Start it and enable it on reboot:
sudo systemctl start ollama
sudo systemctl enable ollama

# Verify Ollama is listening
curl http://localhost:11434
# Expected output: Ollama is running
```

### Pull the llama3.2 model

```bash
# This downloads ~2 GB — takes a few minutes on a typical EC2 connection
ollama pull llama3.2

# Verify the model is ready
ollama run llama3.2 "Reply with one word: ready"
# Expected output: ready  (or similar short response)
```

The model is cached in `/root/.ollama/models/` and persists across reboots.

---

## 6. Clone the Repository

```bash
# Install git if not present (AL2023 includes it; Ubuntu may need it)
sudo dnf install -y git    # Amazon Linux 2023
# sudo apt-get install -y git  # Ubuntu

git clone https://github.com/your-username/rag-document-qa-chatbot.git
cd rag-document-qa-chatbot
```

---

## 7. Build the Docker Image

The image bundles Flask, Gunicorn, sentence-transformers, and ChromaDB. It
does **not** include Ollama or the LLM weights.

```bash
# This takes 5–15 minutes on first build (downloads torch + sentence-transformers)
docker build -t rag-chatbot .

# Confirm the image was built
docker images rag-chatbot
```

---

## 8. Prepare Runtime Directories

ChromaDB and uploaded PDFs are stored on the host and mounted into the
container. This means data survives container restarts and image upgrades.

```bash
mkdir -p ~/rag-data/uploads ~/rag-data/chroma_db
```

---

## 9. Generate a Secret Key

Flask uses `SECRET_KEY` to sign session cookies. Generate a strong random key
before running in production:

```bash
python3 -c "import secrets; print(secrets.token_hex(32))"
```

Copy the output — you will pass it as an environment variable. **Never commit
a real secret key to source control.**

---

## 10. Run the Container

```bash
docker run -d \
  --name rag-chatbot \
  --restart unless-stopped \
  -p 5000:5000 \
  -e SECRET_KEY="<paste-your-secret-key-here>" \
  -e FLASK_DEBUG=false \
  -e OLLAMA_BASE_URL=http://host.docker.internal:11434 \
  -e CHROMA_DB_PATH=/app/chroma_db \
  -e UPLOAD_FOLDER=/app/uploads \
  -v ~/rag-data/uploads:/app/uploads \
  -v ~/rag-data/chroma_db:/app/chroma_db \
  -v hf-cache:/app/.cache/huggingface \
  --add-host=host.docker.internal:host-gateway \
  rag-chatbot
```

Flag explanations:

| Flag | Purpose |
|---|---|
| `-d` | Run in background (detached) |
| `--restart unless-stopped` | Restart automatically on reboot or crash |
| `-p 5000:5000` | Expose port 5000 to the host |
| `-e SECRET_KEY=...` | Flask session signing key (required in production) |
| `-e FLASK_DEBUG=false` | Disable debug mode |
| `-e OLLAMA_BASE_URL=http://host.docker.internal:11434` | Point the app at the host Ollama process (`host.docker.internal` resolves to the host via `--add-host`) |
| `-v ~/rag-data/uploads:/app/uploads` | Persist uploaded PDFs on the host |
| `-v ~/rag-data/chroma_db:/app/chroma_db` | Persist ChromaDB vector data on the host |
| `-v hf-cache:/app/.cache/huggingface` | Cache the HuggingFace embedding model between runs |
| `--add-host=host.docker.internal:host-gateway` | Allow the container to reach the host on Linux (EC2 runs Linux) |

> **Why `--add-host`?**  
> On Linux, `host.docker.internal` is not resolved automatically (it is only
> available on Docker Desktop). `--add-host=host.docker.internal:host-gateway`
> adds the entry to `/etc/hosts` inside the container, so
> `OLLAMA_BASE_URL=http://host.docker.internal:11434` correctly resolves to
> the EC2 host. Alternatively you can set
> `OLLAMA_BASE_URL=http://172.17.0.1:11434` (the default Docker bridge gateway
> IP), but `--add-host` is more portable.

---

## 11. Verify the Deployment

### 1. Check the container is running

```bash
docker ps
# Should show rag-chatbot with status "Up ... (healthy)"
```

### 2. Check the health endpoint from the instance

```bash
curl http://localhost:5000/api/health
# Expected: {"status":"healthy"}
```

### 3. Check from your browser

Navigate to:

```
http://<your-instance-public-ip>:5000
```

You should see the RAG Document Q&A Chatbot UI.

### 4. Check container logs

```bash
docker logs rag-chatbot
# Should show Gunicorn workers starting, no errors
```

### 5. Smoke-test the full RAG pipeline

```bash
# Upload a small PDF
curl -s -X POST http://localhost:5000/api/upload \
  -F "file=@/path/to/sample.pdf" | python3 -m json.tool
# Expected: {"message": "Document processed successfully", "chunks_processed": N}

# Ask a question (Ollama must be running and llama3.2 must be pulled)
curl -s -X POST http://localhost:5000/api/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the main topic of this document?"}' \
  | python3 -m json.tool
# Expected: {"answer": "...", "sources": [...]}
```

---

## 12. Environment Variable Reference

All configuration is passed at container startup via `-e` flags or an
`--env-file`. **Do not bake secrets into the image or commit them to git.**

| Variable | EC2 Value | Description |
|---|---|---|
| `SECRET_KEY` | Long random hex string | Flask session signing key |
| `FLASK_DEBUG` | `false` | Disable debug mode in production |
| `OLLAMA_BASE_URL` | `http://host.docker.internal:11434` | Reach host Ollama via `--add-host` |
| `OLLAMA_MODEL` | `llama3.2` | Model name (must be pulled) |
| `CHROMA_DB_PATH` | `/app/chroma_db` | Absolute path inside the container |
| `UPLOAD_FOLDER` | `/app/uploads` | Absolute path inside the container |
| `HF_HOME` | `/app/.cache/huggingface` | HuggingFace model cache (set in Dockerfile) |
| `MAX_UPLOAD_SIZE_MB` | `50` | Maximum PDF upload size |
| `TOP_K` | `4` | Retrieval chunks per query |

Optional: store variables in a file and pass `--env-file /path/to/.env` to
`docker run`. Make sure the file is **not** committed to git (add it to
`.gitignore`).

---

## 13. Managing the Application

### View logs

```bash
docker logs -f rag-chatbot          # follow live
docker logs --tail 100 rag-chatbot  # last 100 lines
```

### Restart

```bash
docker restart rag-chatbot
```

### Stop and remove

```bash
docker stop rag-chatbot
docker rm rag-chatbot
```

### Upgrade (new image build)

```bash
# Pull the latest code
git pull

# Rebuild the image
docker build -t rag-chatbot .

# Stop the old container
docker stop rag-chatbot && docker rm rag-chatbot

# Start with the new image (same run command as above)
docker run -d --name rag-chatbot --restart unless-stopped ...
```

Your ChromaDB data and uploaded files are safe — they live on the host in
`~/rag-data/`, not inside the container.

### Check Ollama

```bash
systemctl status ollama             # service status
curl http://localhost:11434         # health
ollama list                         # see pulled models
journalctl -u ollama -f             # follow Ollama logs
```

---

## 14. Common Issues

| Symptom | Likely cause | Fix |
|---|---|---|
| `/api/ask` returns 503 | Ollama not running | `sudo systemctl start ollama` |
| `/api/ask` returns 503 | llama3.2 not pulled | `ollama pull llama3.2` |
| Container exits immediately | Secret key or env var issue | `docker logs rag-chatbot` |
| `curl: (7) Failed to connect` on port 5000 | Security group missing port 5000 rule | Add inbound TCP 5000 in the AWS console |
| Upload returns 500 on first request | HuggingFace model download failed | Check internet access; `docker logs rag-chatbot` |
| ChromaDB data lost after restart | Volume not mounted | Confirm `-v ~/rag-data/chroma_db:/app/chroma_db` in run command |
| "permission denied" on `/app/chroma_db` | Host dir owned by root | `sudo chown -R 1001:1001 ~/rag-data` |

---

## 15. Security Considerations

This is a portfolio/demo deployment. For a real production deployment you
would also:

- Put the app behind **nginx** or an **Application Load Balancer** with TLS
  (port 443) instead of exposing port 5000 directly.
- Rotate `SECRET_KEY` and store it in **AWS Secrets Manager** or **Parameter
  Store** rather than passing it as a plain env var.
- Enable **EC2 Instance Connect** or use a bastion host instead of opening
  SSH from 0.0.0.0/0.
- Enable **EBS encryption** for the root and data volumes.
- Set up **CloudWatch** log shipping from Docker (`--log-driver=awslogs`).
- Restrict Ollama to localhost only (already the case — port 11434 is not
  exposed to the security group).
