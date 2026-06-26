import logging
import time
import schedule

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
)
logger = logging.getLogger("live_bot")

from src.live.executor import LiveExecutor

def main():
    logger.info("Starting Antigravity MTF Live Bot...")
    
    executor = LiveExecutor(model_dir="models", symbol="EURUSD")
    if not executor.connect():
        logger.error("Failed to connect to MT5. Exiting.")
        return

    def job():
        try:
            # Give the MT5 broker 2 seconds to finalize the newly closed M5 candle 
            # and push it to our local terminal before we fetch.
            time.sleep(2)
            if not executor.check_connection():
                logger.error("Failed to reconnect to MT5. Skipping iteration.")
                return
            executor.execute_iteration()
        except Exception as e:
            logger.error("Error during execution iteration: %s", e)

    # Schedule the job to run exactly at the beginning of every 5-minute interval
    # We run it at :00, :05, :10, ..., :55
    for minute in range(0, 60, 5):
        time_str = f":{minute:02d}"
        schedule.every().hour.at(time_str).do(job)
        
    logger.info("Bot scheduled to run every 5 minutes aligned to the clock.")
    logger.info("Waiting for the next 5-minute boundary...")

    try:
        while True:
            schedule.run_pending()
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Bot stopped manually. Disconnecting from MT5.")
    finally:
        executor.disconnect()

if __name__ == "__main__":
    main()
