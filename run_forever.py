import sys
import time
import subprocess
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('bot_monitor.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

def run_bot():
    """Run the trading bot with automatic restart on failure"""
    while True:
        try:
            logging.info("Starting trading bot...")
            # Start the bot as a subprocess
            process = subprocess.Popen([sys.executable, "main.py"])
            
            # Wait for the process to complete
            process.wait()
            
            # If we get here, the process has ended
            exit_code = process.returncode
            logging.warning(f"Bot process ended with exit code {exit_code}")
            
            # Wait before restarting
            logging.info("Waiting 60 seconds before restart...")
            time.sleep(60)
            
        except Exception as e:
            logging.error(f"Error running bot: {str(e)}")
            logging.info("Waiting 60 seconds before retry...")
            time.sleep(60)
            continue

if __name__ == "__main__":
    logging.info("Bot monitor started")
    run_bot()
