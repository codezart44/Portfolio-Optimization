echo "> exporting fred api key"
source scripts/env.sh

echo "> setting up data folders"
source scripts/folders.sh

echo "> downloading return data"
python scripts/download.py

echo "> processing risk models"
python scripts/risk.py
