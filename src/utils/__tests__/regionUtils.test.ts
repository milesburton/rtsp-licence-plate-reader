import { describe, it, expect } from 'vitest';
import { findRegions, calculateOverlap, filterOverlappingRegions } from '../regionUtils';
import type { DetectionConfig, Region } from '../types';

describe('regionUtils', () => {
  it('should detect a single region correctly', () => {
    // Example: Simple 5x5 image with a single black region
    const width = 5;
    const height = 5;
    const data = Buffer.from([
      255, 255, 255, 255, 255,
      255, 0,   0,   255, 255,
      255, 0,   0,   255, 255,
      255, 255, 255, 255, 255,
      255, 255, 255, 255, 255,
    ]);

    const config: DetectionConfig = {
      minArea: 1,
      maxArea: 10,
      minAspectRatio: 0.5,
      maxAspectRatio: 2,
    };

    const regions = findRegions(data, width, height, config);
    expect(regions.length).toBe(1);
    expect(regions[0]).toEqual({ x: 1, y: 1, width: 2, height: 2 });
  });

  it('should calculate overlap correctly', () => {
    const r1: Region = { x: 0, y: 0, width: 4, height: 4 };
    const r2: Region = { x: 2, y: 2, width: 4, height: 4 };

    const overlap = calculateOverlap(r1, r2);
    // Overlapping area is 2x2 = 4
    // Smaller area is 4x4 = 16
    expect(overlap).toBeCloseTo(0.25);
  });

  it('should filter overlapping regions correctly', () => {
    const regions: Region[] = [
      { x: 0, y: 0, width: 4, height: 4 }, // Region 1
      { x: 2, y: 2, width: 4, height: 4 }, // Region 2 overlaps with Region 1
      { x: 5, y: 5, width: 2, height: 2 }, // Region 3 no overlap
    ];

    const filtered = filterOverlappingRegions(regions);
    expect(filtered.length).toBe(2);
    expect(filtered).toContainEqual({ x: 0, y: 0, width: 4, height: 4 });
    expect(filtered).toContainEqual({ x: 5, y: 5, width: 2, height: 2 });
  });

  it('should handle multiple non-overlapping regions', () => {
    const width = 10;
    const height = 10;
    const data = Buffer.alloc(width * height, 255);
    // Create two separate regions
    // Region 1
    data[11] = 0;
    data[12] = 0;
    data[21] = 0;
    data[22] = 0;
    // Region 2
    data[71] = 0;
    data[72] = 0;
    data[81] = 0;
    data[82] = 0;

    const config: DetectionConfig = {
      minArea: 1,
      maxArea: 10,
      minAspectRatio: 0.5,
      maxAspectRatio: 2,
    };

    const regions = findRegions(data, width, height, config);
    expect(regions.length).toBe(2);
    expect(regions).toContainEqual({ x: 1, y: 1, width: 1, height: 1 });
    expect(regions).toContainEqual({ x: 1, y: 7, width: 1, height: 1 });
  });

  it('should ignore regions smaller than minArea', () => {
    const width = 5;
    const height = 5;
    const data = Buffer.from([
      255, 255, 255, 255, 255,
      255, 0,   255, 255, 255,
      255, 255, 255, 255, 255,
      255, 255, 0,   255, 255,
      255, 255, 255, 255, 255,
    ]);

    const config: DetectionConfig = {
      minArea: 2,
      maxArea: 10,
      minAspectRatio: 0.5,
      maxAspectRatio: 2,
    };

    const regions = findRegions(data, width, height, config);
    expect(regions.length).toBe(0);
  });

  it('should ignore regions larger than maxArea', () => {
    const width = 6;
    const height = 6;
    const data = Buffer.alloc(width * height, 255);
    // Create a large region
    for (let i = 0; i < 4; i++) {
      for (let j = 0; j < 4; j++) {
        data[i * width + j] = 0;
      }
    }

    const config: DetectionConfig = {
      minArea: 1,
      maxArea: 10, // The region area is 16
      minAspectRatio: 0.5,
      maxAspectRatio: 2,
    };

    const regions = findRegions(data, width, height, config);
    expect(regions.length).toBe(0);
  });
});
