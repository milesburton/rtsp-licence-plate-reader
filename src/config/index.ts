import { env } from "bun";
import { logger } from "../utils/logger";
import { EMOJIS } from "../utils/logger";
import * as path from "path";

export function validateConfig() {
    const requiredVars = ['RTSP_URL'];
    const missingVars = requiredVars.filter(varName => !env[varName]);

    if (missingVars.length > 0) {
        logger.error(`${EMOJIS.NO_PLATE} Missing required environment variables: ${missingVars.join(', ')}`);
        process.exit(1);
    }

    const numericVars = ['FPS', 'FRAME_WIDTH', 'FRAME_HEIGHT', 'MAX_RETRIES', 'RETRY_DELAY',
        'MIN_VEHICLE_AREA', 'MAX_VEHICLE_AREA', 'MIN_PERSON_AREA', 'MAX_PERSON_AREA'];

    numericVars.forEach(varName => {
        const value = env[varName];
        if (value && isNaN(Number(value))) {
            logger.error(`${EMOJIS.NO_PLATE} Invalid numeric value for ${varName}: ${value}`);
            process.exit(1);
        }
    });
}

export const CONFIG = {
    RTSP_URL: env.RTSP_URL,
    FPS: parseInt(env.FPS || '15'),
    FRAME_WIDTH: parseInt(env.FRAME_WIDTH || '1920'),
    FRAME_HEIGHT: parseInt(env.FRAME_HEIGHT || '1080'),
    MAX_RETRIES: parseInt(env.MAX_RETRIES || '3'),
    RETRY_DELAY: parseInt(env.RETRY_DELAY || '5000'),
    DEBUG_MODE: env.DEBUG_MODE === 'true',
    MIN_VEHICLE_AREA: parseInt(env.MIN_VEHICLE_AREA || '5000'),
    MAX_VEHICLE_AREA: parseInt(env.MAX_VEHICLE_AREA || '120000'),
    MIN_PERSON_AREA: parseInt(env.MIN_PERSON_AREA || '5000'),
    MAX_PERSON_AREA: parseInt(env.MAX_PERSON_AREA || '50000'),
    PLATE_PATTERNS: {
        UK: /^[A-Z]{2}[0-9]{2}[A-Z]{3}$/,
        US: /^[A-Z0-9]{5,8}$/,
        EU: /^[A-Z]{1,2}[0-9]{1,4}[A-Z]{1,2}$/
    },
    FRAME_QUEUE_SIZE: parseInt(env.FRAME_QUEUE_SIZE || '30'),
};

export const TEMP_DIR = path.resolve(import.meta.dir, '../temp');
