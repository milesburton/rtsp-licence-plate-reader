import * as tf from '@tensorflow/tfjs-node';
import * as fs from 'fs';
import * as path from 'path';
import sharp from 'sharp';
import { logger } from '../utils/logger';
import { CONFIG, TEMP_DIR } from '../config';
import { detectPeople, detectVehicles, performOCR } from './detection';
import { EMOJIS } from '../utils/logger';

export async function processFrame(buffer: Buffer) {
    try {
        logger.info(`${EMOJIS.FRAME_PROCESS} Processing frame (size: ${buffer.length} bytes)`);

        // Convert incoming buffer to compatible JPEG format
        const processedBuffer = await sharp(buffer)
            .jpeg()
            .toBuffer();

        // Decode the processed image
        const tensor = tf.node.decodeJpeg(buffer) as tf.Tensor3D;
        logger.debug(`${EMOJIS.DEBUG} Created tensor with shape: ${tensor.shape} and dtype: ${tensor.dtype}`);

        const tempFramePath = path.join(TEMP_DIR, `frame_${Date.now()}.jpg`);
        await fs.promises.writeFile(tempFramePath, processedBuffer);
        logger.debug(`${EMOJIS.DEBUG} Frame written to ${tempFramePath}`);

        const people = await detectPeople(tempFramePath);
        if (people.length > 0) {
            const peopleConfidences = people
                .map(p => `👤 Person at (${p.x}, ${p.y}): ${p.confidence.toFixed(1)}% confidence`)
                .join('\n');
            logger.info(`${EMOJIS.PEOPLE_DETECT} Detected ${people.length} people in frame:\n${peopleConfidences}`);
        }

        const vehicles = await detectVehicles(tempFramePath);
        if (vehicles.length > 0) {
            const vehicleDetails = vehicles
                .map(v => `🚗 ${v.type} at (${v.x}, ${v.y}): ${v.confidence.toFixed(1)}% confidence`)
                .join('\n');
            logger.info(`${EMOJIS.VEHICLE_DETECT} Detected ${vehicles.length} vehicles in frame:\n${vehicleDetails}`);

            for (const vehicle of vehicles) {
                try {
                    const vehicleTensor = tf.slice(tensor, [vehicle.y, vehicle.x, 0], [vehicle.height, vehicle.width, 3]) as tf.Tensor3D;
                    const vehicleBuffer = await tf.node.encodeJpeg(vehicleTensor);
                    const vehiclePath = path.join(TEMP_DIR, `vehicle_${Date.now()}.jpg`);
                    await fs.promises.writeFile(vehiclePath, vehicleBuffer);
                    logger.debug(`${EMOJIS.DEBUG} Vehicle region written to ${vehiclePath}`);

                    const plates = await performOCR(vehiclePath);
                    if (plates.length > 0) {
                        const plateConfidences = plates
                            .map(p => `🚘 Detected plate: ${p.text} (${(p.confidence * 100).toFixed(1)}% confidence)`)
                            .join('\n');
                        logger.info(`${EMOJIS.PLATE_DETECT} Detected license plates:\n${plateConfidences}`);
                    }

                    await fs.promises.unlink(vehiclePath);
                    logger.debug(`${EMOJIS.DEBUG} Vehicle region file deleted: ${vehiclePath}`);
                    vehicleTensor.dispose();
                } catch (error) {
                    logger.error(`${EMOJIS.NO_PLATE} Error processing vehicle region:`, error);
                    if (error instanceof Error) {
                        logger.error(`${EMOJIS.NO_PLATE} Stack trace:`, error.stack);
                    }
                }
            }
        }

        await fs.promises.unlink(tempFramePath);
        logger.debug(`${EMOJIS.DEBUG} Frame file deleted: ${tempFramePath}`);
        tensor.dispose();
    } catch (error) {
        logger.error(`${EMOJIS.NO_PLATE} Error processing frame:`, error);
        if (error instanceof Error) {
            logger.error(`${EMOJIS.NO_PLATE} Stack trace:`, error.stack);
        }
    }
}