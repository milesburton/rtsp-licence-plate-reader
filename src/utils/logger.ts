import winston from 'winston';
import chalk from 'chalk';
import * as path from 'path';
import * as fs from 'fs';

const customFormat = winston.format.printf(({ level, message, timestamp }) => {
    let emoji = '';
    let color = chalk.white;

    switch (level) {
        case 'info':
            emoji = 'ℹ️';
            color = chalk.blue;
            break;
        case 'warn':
            emoji = '⚠️';
            color = chalk.yellow;
            break;
        case 'error':
            emoji = '❌';
            color = chalk.red;
            break;
        case 'debug':
            emoji = '🐛';
            color = chalk.green;
            break;
        default:
            emoji = '🔍';
            break;
    }

    return `${color(`${emoji} [${timestamp}] ${level.toUpperCase()}:`)} ${message}`;
});

const logDir = path.resolve(__dirname, '../logs');

if (!fs.existsSync(logDir)) {
    try {
        fs.mkdirSync(logDir, { recursive: true });
        console.log(`Created log directory at ${logDir}`);
    } catch (error) {
        console.error(`Failed to create log directory at ${logDir}:`, error);
        process.exit(1);
    }
}

export const logger = winston.createLogger({
    level: 'info',
    format: winston.format.combine(
        winston.format.timestamp({ format: 'YYYY-MM-DD HH:mm:ss' }),
        winston.format.errors({ stack: true }),
        customFormat
    ),
    transports: [
        new winston.transports.Console(),
        new winston.transports.File({ filename: path.join(logDir, 'error.log'), level: 'error' }),
        new winston.transports.File({ filename: path.join(logDir, 'app.log') })
    ],
});

export const EMOJIS = {
    INIT: '🚀',
    DIR_CHECK: '📂',
    RTSP_CONNECT: '📡',
    FFMPEG: '🎥',
    FRAME_WRITE: '🖼️',
    FRAME_PROCESS: '🔍',
    PEOPLE_DETECT: '👤',
    VEHICLE_DETECT: '🚗',
    PLATE_DETECT: '🚘',
    NO_PLATE: '🚫',
    DEBUG: '🐞',
};
