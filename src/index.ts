import * as tf from '@tensorflow/tfjs-node';
import { logger } from './utils/logger';
import { CONFIG, validateConfig } from './config';
import { startStreamProcessing } from './services/stream';
import { cleanup } from './services/cleanup';
import { initialise } from './services/detection';
import { EMOJIS } from './utils/logger';

// Validate configuration
validateConfig();

const HEARTBEAT_INTERVAL = 10000; // 10 seconds

// Handle unhandled promise rejections
process.on('unhandledRejection', (reason, promise) => {
    logger.error(`${EMOJIS.NO_PLATE} Unhandled Rejection at:`, promise);
    if (reason instanceof Error) {
        logger.error(`${EMOJIS.NO_PLATE} Stack trace:`, reason.stack);
    } else {
        logger.error(`${EMOJIS.NO_PLATE} Reason:`, reason);
    }
    process.exit(1);
});

// Handle process exit
process.on('exit', (code) => {
    logger.info(`${EMOJIS.NO_PLATE} Process is exiting with code: ${code}`);
});

// Handle uncaught exceptions
process.on('uncaughtException', (error) => {
    logger.error(`${EMOJIS.NO_PLATE} Uncaught exception:`, error);
    logger.error(`${EMOJIS.NO_PLATE} Stack trace:`, error.stack);
    cleanup();
    process.exit(1);
});

// Graceful shutdown handling
process.on('SIGINT', cleanup);
process.on('SIGTERM', cleanup);

// Heartbeat to indicate the application is running
setInterval(() => {
    logger.info(`${EMOJIS.INIT} Heartbeat: Application is running...`);
}, HEARTBEAT_INTERVAL);

// Memory statistics logging
setInterval(() => {
    const memInfo = tf.memory();
    logger.info(
        `${EMOJIS.DEBUG} TensorFlow Memory Stats:
        Active Tensors: ${memInfo.numTensors}
        Data Buffers: ${memInfo.numDataBuffers}
        Memory Used: ${(memInfo.numBytes / 1024 / 1024).toFixed(2)} MB
        Memory State: ${memInfo.unreliable ? 'Unreliable' : 'Reliable'}`
    );
}, HEARTBEAT_INTERVAL);

function logConfiguration(config: typeof CONFIG) {
    logger.info(`${EMOJIS.INIT} Application configuration:`);
    logger.info(`${EMOJIS.INIT} ----------------------------------------`);
    logger.info(`${EMOJIS.INIT} FPS: ${config.FPS}`);
    logger.info(`${EMOJIS.INIT} Frame Width: ${config.FRAME_WIDTH}`);
    logger.info(`${EMOJIS.INIT} Frame Height: ${config.FRAME_HEIGHT}`);
    logger.info(`${EMOJIS.INIT} Max Retries: ${config.MAX_RETRIES}`);
    logger.info(`${EMOJIS.INIT} Retry Delay: ${config.RETRY_DELAY} ms`);
    logger.info(`${EMOJIS.INIT} Debug Mode: ${config.DEBUG_MODE ? 'Enabled' : 'Disabled'}`);
    logger.info(`${EMOJIS.INIT} Min Vehicle Area: ${config.MIN_VEHICLE_AREA}`);
    logger.info(`${EMOJIS.INIT} Max Vehicle Area: ${config.MAX_VEHICLE_AREA}`);
    logger.info(`${EMOJIS.INIT} Min Person Area: ${config.MIN_PERSON_AREA}`);
    logger.info(`${EMOJIS.INIT} Max Person Area: ${config.MAX_PERSON_AREA}`);
    logger.info(`${EMOJIS.INIT} Frame Queue Size: ${config.FRAME_QUEUE_SIZE}`);
    logger.info(`${EMOJIS.INIT} ----------------------------------------`);
}

// Start the application
(async () => {
    await initialise();
    logConfiguration(CONFIG);
    startStreamProcessing();
})();
