path_log="logs/run_$(date +%Y%m%d).log"

echo "Running Script..." >> "$path_log" 2>&1

python main.py -cfg config/config_cnn.yml >> "$path_log" 2>&1

echo "Run Finished!" >> "$path_log" 2>&1