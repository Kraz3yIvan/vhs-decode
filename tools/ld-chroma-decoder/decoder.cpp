/************************************************************************

    decoder.cpp

    ld-chroma-decoder - Colourisation filter for ld-decode
    Copyright (C) 2019-2021 Adam Sampson

    This file is part of ld-decode-tools.

    ld-chroma-decoder is free software: you can redistribute it and/or
    modify it under the terms of the GNU General Public License as
    published by the Free Software Foundation, either version 3 of the
    License, or (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.

************************************************************************/

#include "decoder.h"

#include "decoderpool.h"

qint32 Decoder::getLookBehind() const
{
    return 0;
}

qint32 Decoder::getLookAhead() const
{
    return 0;
}

DecoderThread::DecoderThread(QAtomicInt& _abort, DecoderPool& _decoderPool, QObject *parent)
    : QThread(parent), abort(_abort), decoderPool(_decoderPool), outputWriter(_decoderPool.getOutputWriter())
{
}

void DecoderThread::run()
{
    // V4: If chunk mode is active, take the chunk-parallel path.
    if (decoderPool.isChunkMode()) {
        runChunk();
        return;
    }

    // --- Original sequential batch path (1D/2D, or single-thread) ---
    QVector<SourceField> inputFields;
    QVector<ComponentFrame> componentFrames;
    QVector<OutputFrame> outputFrames;

    while (!abort) {
        // Get the next batch of fields to process
        qint32 startFrameNumber, startIndex, endIndex;
        if (!decoderPool.getInputFrames(startFrameNumber, inputFields, startIndex, endIndex)) {
            // No more input frames -- exit
            break;
        }

        // Adjust the temporary arrays to the right size
        const qint32 numFrames = (endIndex - startIndex) / 2;
        componentFrames.resize(numFrames);
        outputFrames.resize(numFrames);

        // Decode the fields to component frames
        decodeFrames(inputFields, startIndex, endIndex, componentFrames);

        // Convert the component frames to the output format
        for (qint32 i = 0; i < numFrames; i++) {
            outputWriter.convert(componentFrames[i], outputFrames[i]);
        }

        // Write the frames to the output file
        if (!decoderPool.putOutputFrames(startFrameNumber, outputFrames)) {
            abort = true;
            break;
        }
    }
}

// =========================================================================
// V4: Chunk-parallel processing
//
// Each thread:
//   1. Atomically claims a chunk (disjoint frame range)
//   2. Opens its OWN SourceVideo (SourceVideo is not thread-safe)
//   3. Processes its chunk in small batches (bounded memory)
//   4. Guard frames at chunk boundaries are decoded for temporal context
//      but their output is discarded
//
// Memory is bounded: each thread holds at most BATCH_SIZE frames at a
// time (~16 frames ≈ 16 MB fields + 38 MB FrameBuffers per thread),
// regardless of chunk size.
// =========================================================================
void DecoderThread::runChunk()
{
    DecoderPool::ChunkInfo chunk;
    if (!decoderPool.getChunkAssignment(chunk)) {
        return;  // More threads than chunks — nothing to do
    }

    // Open a private SourceVideo for this thread.
    // SourceVideo is NOT thread-safe (shared file position + output buffer),
    // so each chunk thread must have its own instance.
    const auto videoParameters = decoderPool.getMetaData().getVideoParameters();
    const qint32 fieldLength = videoParameters.fieldWidth * videoParameters.fieldHeight;

    SourceVideo sourceVideo;
    if (!sourceVideo.open(decoderPool.getInputFileName(), fieldLength)) {
        qCritical() << "V4: Thread" << QThread::currentThreadId()
                     << "— failed to open source video for chunk"
                     << chunk.chunkStart << "-" << chunk.chunkEnd;
        abort = true;
        return;
    }

    const qint32 lookBehind = decoderPool.getDecoderLookBehind();
    const qint32 lookAhead = decoderPool.getDecoderLookAhead();

    // Process the chunk in small batches to bound memory usage.
    static constexpr qint32 BATCH_SIZE = 16;
    qint32 frameNumber = chunk.loadStart;

    QVector<SourceField> fields;
    QVector<ComponentFrame> componentFrames;
    QVector<OutputFrame> outputFrames;

    while (frameNumber < chunk.loadEnd && !abort) {
        const qint32 batchFrames = qMin(static_cast<qint32>(BATCH_SIZE),
                                        chunk.loadEnd - frameNumber);

        // Load this batch's fields (with decoder lookBehind/lookAhead)
        qint32 startIndex, endIndex;
        SourceField::loadFields(sourceVideo, decoderPool.getMetaData(),
                                frameNumber, batchFrames,
                                lookBehind, lookAhead,
                                fields, startIndex, endIndex);

        const qint32 numFrames = (endIndex - startIndex) / 2;
        componentFrames.resize(numFrames);

        // Decode this batch through the full pipeline
        decodeFrames(fields, startIndex, endIndex, componentFrames);

        if (abort) break;

        // Output only frames within the real range [chunkStart, chunkEnd).
        // Guard frames (outside this range) are decoded but discarded.
        const qint32 batchOutputStart = qMax(frameNumber, chunk.chunkStart);
        const qint32 batchOutputEnd = qMin(frameNumber + batchFrames, chunk.chunkEnd);

        if (batchOutputStart < batchOutputEnd) {
            const qint32 outputCount = batchOutputEnd - batchOutputStart;
            outputFrames.resize(outputCount);

            const qint32 offsetInBatch = batchOutputStart - frameNumber;
            for (qint32 i = 0; i < outputCount; i++) {
                outputWriter.convert(componentFrames[offsetInBatch + i], outputFrames[i]);
            }

            if (!decoderPool.putOutputFrames(batchOutputStart, outputFrames)) {
                abort = true;
                break;
            }
        }

        frameNumber += batchFrames;
    }

    sourceVideo.close();
}
