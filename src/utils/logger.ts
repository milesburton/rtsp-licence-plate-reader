import winston from 'winston';

const { combine, timestamp, printf, colorize } = winston.format;

const logFormat = printf(({ level, message, timestamp }) => {
    let emoji = '';
    switch (level) {
        case 'info':
            emoji = 'ℹ️';
            break;
        case 'error':
            emoji = '❌';
            break;
        case 'warn':
            emoji = '⚠️';
            break;
        case 'debug':
            emoji = '🐛';
            break;
        default:
            emoji = '💬';
    }
    return `${timestamp} ${emoji} [${level.toUpperCase()}]: ${message}`;
});

const logger = winston.createLogger({
    level: 'info',
    format: combine(
        colorize(),
        timestamp({ format: 'YYYY-MM-DD HH:mm:ss' }),
        logFormat
    ),
    transports: [
        new winston.transports.Console(),
        new winston.transports.File({ filename: 'logs/error.log', level: 'error' }),
        new winston.transports.File({ filename: 'logs/app.log' })
    ],
});

export default logger;
