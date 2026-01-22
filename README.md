# simpson-pipeline-backend
Simpson Pipeline Backend — Run Guide

This backend service:

• accepts PDFs via API
• runs AI pipeline phases
• stores results in MongoDB
• saves JSON outputs to disk
• supports cron batch jobs
• exposes Swagger API

📦 Requirements

System:

Linux (tested on Kali)

Python 3.11+

Docker

Git

Python packages installed from:

requirements.txt

🛠️ Setup
1️⃣ Clone Repository
git clone https://github.com/<your-username>/simpson-pipeline-backend.git
cd simpson-pipeline-backend

2️⃣ Create Virtual Env (recommended)
python -m venv venv
source venv/bin/activate


OR pyenv users:

pyenv shell 3.11.7

3️⃣ Install Python Dependencies
python -m pip install -r requirements.txt

🐳 MongoDB (Docker)

MongoDB runs inside Docker.

Start Mongo
sudo docker run -d \
  --name simpson-mongo \
  -p 27017:27017 \
  -v simpson-mongo-data:/data/db \
  mongo:7

Verify Mongo
sudo docker ps
sudo docker exec -it simpson-mongo mongosh


Inside shell:

show dbs


Exit with Ctrl+D.

▶️ Run Backend API

From project root:

python -m uvicorn app:app --reload


Open Swagger:

👉 http://127.0.0.1:8000/docs

📤 Run Pipeline via API

In Swagger:

POST /pipeline/run

Upload PDF → Execute.

Response:

{
  "run_id": "...",
  "status": "started"
}

📊 Check Run Status

GET:

/pipeline/{run_id}

📁 File Outputs

Uploaded PDFs:

uploads/


JSON results:

outputs/<run_id>.json

🗄️ Mongo Inspection
sudo docker exec -it simpson-mongo mongosh
use simpson_pipeline
db.runs.find().pretty()

⏰ Cron Batch Runs

Cron processes PDFs placed into:

cron_inputs/


Processed PDFs move to:

cron_archive/

Run Cron Manually (test)
python cron_runner.py

Add Cron Job
crontab -e


Add:

0 2 * * * cd /home/kali/simpson-pipeline-backend && /home/kali/.pyenv/shims/python cron_runner.py >> cron.log 2>&1


Runs daily at 2 AM.

🧹 Git Ignore

These folders are ignored:

uploads/
outputs/
cron.log
__pycache__/
.env

🧠 Architecture
simpson-pipeline-backend/
│
├── app.py
├── cron_runner.py
├── requirements.txt
├── uploads/
├── outputs/
├── cron_inputs/
├── cron_archive/
│
└── pipeline/
    ├── runner.py
    ├── phase1_v3.py
    ├── phase2_v3.py
    ├── phase3_v4.py
    ├── phase4_v3.py
    ├── phase5_v2.py
