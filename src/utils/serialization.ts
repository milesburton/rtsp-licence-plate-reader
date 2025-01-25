/**
 * Converts a Node.js Buffer to an ArrayBuffer.
 *
 * @param buffer - The Buffer to convert.
 * @returns The resulting ArrayBuffer.
 */
export function bufferToArrayBuffer(buffer: Buffer): ArrayBuffer {
    return buffer.buffer.slice(buffer.byteOffset, buffer.byteOffset + buffer.byteLength);
}