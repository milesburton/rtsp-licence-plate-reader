import * as tf from '@tensorflow/tfjs-node';
import * as fs from 'fs';
import { logger } from '../utils/logger';
import { TEMP_DIR } from '../config';
import { EMOJIS } from '../utils/logger';

export let isShuttingDown = false;

export async function cleanup() {
    if (isShuttingDown) return;
    isShuttingDown = true;

    logger.info(`${EMOJIS.NO_PLATE} Cleaning up...`);

    try {
        if (fs.existsSync(TEMP_DIR)) {
            await fs.promises.rm(TEMP_DIR, { recursive: true, force: true });
            logger.info(`${EMOJIS.INIT} Temporary directory deleted: ${TEMP_DIR}`);
        }
    } catch (error) {
        logger.error(`${EMOJIS.NO_PLATE} Error cleaning up temporary directory:`, error);
    }

    tf.disposeVariables();
    logger.info(`${EMOJIS.INIT} TensorFlow variables disposed`);

    process.exit(0);
}
