import type { Region, DetectionConfig } from './types';

/**
 * Finds regions within the image data based on the provided configuration.
 *
 * @param data - The raw image data as a Buffer.
 * @param width - The width of the image.
 * @param height - The height of the image.
 * @param config - Configuration for detection thresholds.
 * @returns An array of detected regions.
 */
export function findRegions(
    data: Buffer,
    width: number,
    height: number,
    config: DetectionConfig
): Region[] {
    const visited = new Set<number>();
    const regions: Region[] = [];

    for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
            const pos = y * width + x;
            if (data[pos] === 0 && !visited.has(pos)) { // Assuming black pixels represent potential regions
                const region = floodFill(data, width, height, x, y, visited);
                const bounds = getBoundingBox(region, width);

                const aspectRatio = bounds.width / bounds.height;
                const area = bounds.width * bounds.height;

                if (
                    area >= config.minArea &&
                    area <= config.maxArea &&
                    aspectRatio >= config.minAspectRatio &&
                    aspectRatio <= config.maxAspectRatio
                ) {
                    regions.push(bounds);
                }
            }
        }
    }

    return filterOverlappingRegions(regions);
}

/**
 * Performs a flood fill algorithm to identify connected regions.
 *
 * @param data - The raw image data as a Buffer.
 * @param width - The width of the image.
 * @param height - The height of the image.
 * @param x - The starting x-coordinate.
 * @param y - The starting y-coordinate.
 * @param visited - A set to keep track of visited positions.
 * @returns An array of positions belonging to the detected region.
 */
function floodFill(
    data: Buffer,
    width: number,
    height: number,
    x: number,
    y: number,
    visited: Set<number>
): number[] {
    const region: number[] = [];
    const stack: { x: number; y: number }[] = [{ x, y }];

    while (stack.length > 0) {
        const { x: currentX, y: currentY } = stack.pop()!;
        const pos = currentY * width + currentX;

        if (
            currentX < 0 ||
            currentX >= width ||
            currentY < 0 ||
            currentY >= height ||
            visited.has(pos) ||
            data[pos] !== 0
        ) {
            continue;
        }

        visited.add(pos);
        region.push(pos);

        // Add neighboring pixels to the stack
        stack.push(
            { x: currentX + 1, y: currentY },
            { x: currentX - 1, y: currentY },
            { x: currentX, y: currentY + 1 },
            { x: currentX, y: currentY - 1 }
        );
    }

    return region;
}

/**
 * Calculates the bounding box for a given set of positions.
 *
 * @param region - An array of positions belonging to the region.
 * @param width - The width of the image.
 * @returns The bounding box as a Region object.
 */
export function getBoundingBox(region: number[], width: number): Region {
    let minX = Infinity,
        minY = Infinity,
        maxX = -Infinity,
        maxY = -Infinity;

    for (const pos of region) {
        const x = pos % width;
        const y = Math.floor(pos / width);
        minX = Math.min(minX, x);
        maxX = Math.max(maxX, x);
        minY = Math.min(minY, y);
        maxY = Math.max(maxY, y);
    }

    return {
        x: minX,
        y: minY,
        width: maxX - minX,
        height: maxY - minY,
    };
}

/**
 * Filters out overlapping regions based on a specified overlap threshold.
 *
 * @param regions - An array of detected regions.
 * @returns A filtered array of regions with minimal overlaps.
 */
export function filterOverlappingRegions(regions: Region[]): Region[] {
    const filtered: Region[] = [];

    for (let i = 0; i < regions.length; i++) {
        let overlapping = false;
        for (let j = 0; j < filtered.length; j++) {
            const overlap = calculateOverlap(regions[i], filtered[j]);
            if (overlap > 0.5) { // Overlap threshold of 50%
                overlapping = true;
                break;
            }
        }
        if (!overlapping) {
            filtered.push(regions[i]);
        }
    }

    return filtered;
}

/**
 * Calculates the overlap ratio between two regions.
 *
 * @param r1 - The first region.
 * @param r2 - The second region.
 * @returns The overlap ratio as a number between 0 and 1.
 */
function calculateOverlap(r1: Region, r2: Region): number {
    const xOverlap = Math.max(
        0,
        Math.min(r1.x + r1.width, r2.x + r2.width) - Math.max(r1.x, r2.x)
    );
    const yOverlap = Math.max(
        0,
        Math.min(r1.y + r1.height, r2.y + r2.height) - Math.max(r1.y, r2.y)
    );
    const overlapArea = xOverlap * yOverlap;
    const r1Area = r1.width * r1.height;
    const r2Area = r2.width * r2.height;
    return overlapArea / Math.min(r1Area, r2Area);
}
