import { env } from 'bun';

interface Config {
    RTSP_URL: string;
    FPS: number;
    FRAME_WIDTH: number;
    FRAME_HEIGHT: number;
    MAX_RETRIES: number;
    RETRY_DELAY: number;
    DEBUG_MODE: boolean;
    MIN_VEHICLE_AREA: number;
    MAX_VEHICLE_AREA: number;
    MIN_PERSON_AREA: number;
    MAX_PERSON_AREA: number;
    PLATE_PATTERNS: { [key: string]: RegExp };
    USE_GPU: boolean;
}

const CONFIG: Config = {
    RTSP_URL: env['RTSP_URL'] || '',
    FPS: parseInt(env['FPS'] || '5'),
    FRAME_WIDTH: parseInt(env['FRAME_WIDTH'] || '1280'),
    FRAME_HEIGHT: parseInt(env['FRAME_HEIGHT'] || '720'),
    MAX_RETRIES: parseInt(env['MAX_RETRIES'] || '3'),
    RETRY_DELAY: parseInt(env['RETRY_DELAY'] || '5000'),
    DEBUG_MODE: env['DEBUG_MODE'] === 'true',
    MIN_VEHICLE_AREA: parseInt(env['MIN_VEHICLE_AREA'] || '10000'),
    MAX_VEHICLE_AREA: parseInt(env['MAX_VEHICLE_AREA'] || '80000'),
    MIN_PERSON_AREA: parseInt(env['MIN_PERSON_AREA'] || '5000'),
    MAX_PERSON_AREA: parseInt(env['MAX_PERSON_AREA'] || '50000'),
    PLATE_PATTERNS: {
        UK: /^[A-Z]{2}[0-9]{2}[A-Z]{3}$/,
        US: /^[A-Z0-9]{5,8}$/,
        EU: /^[A-Z]{1,2}[0-9]{1,4}[A-Z]{1,2}$/
    },
    USE_GPU: env['USE_GPU'] === 'true'
};

if (!CONFIG.RTSP_URL) {
    throw new Error('❌ RTSP_URL environment variable is not set.');
}

export default CONFIG;
