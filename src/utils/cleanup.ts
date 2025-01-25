import logger from './logger';
import fs from 'fs/promises';
import path from 'path';
import { Worker } from 'tesseract.js';

let worker: Worker | null = null; // Ensure this references the actual worker instance

export async function cleanup() {
    try {
        logger.info('🧹 Cleaning up resources...');
        
        if (worker) {
            await worker.terminate();
            logger.info('🔒 OCR Worker terminated.');
        }

        const tempDir = path.join(__dirname, '../../debug_output');
        await fs.rm(tempDir, { recursive: true, force: true });
        logger.info(`🗑️ Removed directory: ${tempDir}`);
        
        process.exit(0);
    } catch (error) {
        logger.error(`❌ Cleanup error: ${error}`);
        process.exit(1);
    }
}
