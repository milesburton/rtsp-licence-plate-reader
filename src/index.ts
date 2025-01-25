import { initialiseWorker, terminateWorker } from './utils/processing';
import logger from './utils/logger';
import { StreamProcessor } from './services/streamProcessor';

/**
 * Main function to initialize the application.
 */
async function main() {
    try {
        logger.info('🚀 Starting RTSP License Plate Reader Application...');
        
        // Initialize the Tesseract OCR Worker
        await initialiseWorker();

        // Initialize and start the Stream Processor
        const streamProcessor = new StreamProcessor();
        streamProcessor.startStream();

        // Handle graceful shutdown
        process.on('SIGINT', async () => {
            logger.info('🛑 Received SIGINT. Shutting down gracefully...');
            await shutdown(streamProcessor);
        });

        process.on('SIGTERM', async () => {
            logger.info('🛑 Received SIGTERM. Shutting down gracefully...');
            await shutdown(streamProcessor);
        });

    } catch (error) {
        logger.error(`❌ Failed to initialize application: ${error}`);
        process.exit(1);
    }
}

/**
 * Handles the shutdown process by terminating workers and cleaning up resources.
 *
 * @param streamProcessor - The instance of StreamProcessor to stop.
 */
async function shutdown(streamProcessor: StreamProcessor) {
    try {
        // Terminate the Stream Processor
        streamProcessor.stopStream();

        // Terminate the Tesseract OCR Worker
        await terminateWorker();

        logger.info('✅ Application shutdown complete.');
        process.exit(0);
    } catch (error) {
        logger.error(`❌ Error during shutdown: ${error}`);
        process.exit(1);
    }
}

main();
